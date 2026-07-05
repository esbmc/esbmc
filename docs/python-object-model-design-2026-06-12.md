# Python Object Model & Class-Handling — Design Note (2026-06-12)

Design pass for §5 #1 of `python-issues-triage-report-2026-06-02.md` (umbrella **#3067**,
with the concrete blockers **#4773/#4117** and **#4796**). This note records the diagnosis,
an evidence-backed design, and a staged plan. **No implementation has started** — this is the
check-in artefact before the build.

---

## 1. Diagnosis — the §5 item is two orthogonal concerns

A code-exploration of the whole class-handling surface (≈12 files) shows that "flow-sensitive
class tracking" actually decomposes into two independent problems that the triage report
conflated:

### 1a. Class logic is scattered (#3067 — a refactor, flips nothing alone)
Class-related logic lives in, at least:
- `python_class.{h,cpp}` — structural parse (methods/attrs/bases).
- `python_class_builder.cpp` — `build`, `get_bases` (flat single-level inheritance merge),
  `add_self_attrs`, `get_members`, `gen_ctor`, `is_typeddict_class`.
- `converter/converter_class.cpp` — `get_attributes_from_self`, the **five-pass**
  `infer_attr_type_from_usage` usage-site scanner, instance-attribute registration.
- `converter/converter_expr.cpp:451-1201` — all attribute reads, nested-attribute chain
  resolution, `flow_class_map_` fallback, `resolve_member_on_base`, the
  `Cannot resolve nested attribute` abort (`:680`).
- `converter/converter_stmt.cpp` — constructor detection + lowering (`:1152`, `:3150-3183`).
- `function_call/builder.cpp` + `function_call/expr.cpp` — method-call symbol-id construction,
  Constructor/ClassMethod/InstanceMethod classification, single-hop base-class method fallback.
- `type_handler.cpp:99` `is_constructor_call`; `json_utils.h` `find_class`/`is_class`
  (flat scans, 20+ call sites); `converter_symbols.cpp:96` `find_function_in_base_classes`.
- Python preprocessor: `class_context_mixin.py`, `core_visitors_mixin.py:456`
  (`instance_class_map`, module-level-only).

There is **no MRO** (no C3 linearisation); inheritance is a one-level left-to-right struct-field
merge, and base-method lookup is single-hop. #3067 asks to centralise all of the above into a
`ClassHandler` with a clear API and unit tests. This improves maintainability and is the natural
home for a proper MRO, but **does not by itself flip any KNOWNBUG**.

### 1b. Instances are stack-allocated (the real KNOWNBUG blocker)
The flow-sensitive *type* tracking is already largely done: Pass 5 of
`infer_attr_type_from_usage` made `github_4117_func_local_attr` CORE, and `flow_class_map_`
handles top-level nested attributes. The remaining failures are **object lifetime**, not typing.

`n1 = Node(1)` lowers to a frontend stack allocation:
```
DECLARE temp : struct Node          // converter_stmt.cpp:3156  code_declt
CALL Node(&temp, 1)                 // converter_stmt.cpp:3178  self = &temp
```
`temp` lives on the **enclosing function's frame**. So `github_4117_function_internal` —
```python
def setup() -> Node:
    n1 = Node(1); n2 = Node(2); n1.next = n2; return n1
n = setup(); assert n.next.value == 2
```
fails with `dereference failure: accessed expired variable pointer 'n2'` (reproduced
2026-06-12). The type of `.next` resolves correctly; the returned pointer is to `setup()`'s
expired frame. CPython heap-allocates, so this is safe at runtime. This same stack-allocation
root underlies `reverse_linked_list` (#4796) and the graph quixbugs (`detect_cycle`,
`depth_first_search`, `topological_ordering`) whose `__ESBMC_get_object_size` diagnostics stem
from class-graph objects.

---

## 2. Evidence — lists already do it right

Lists/dicts/sets do **not** have this problem. A list returned from a function verifies:
```python
def make():
    xs = [1, 2, 3]; return xs
ys = make(); assert ys[1] == 2          # VERIFICATION SUCCESSFUL (2026-06-12)
```
because the list backing store is allocated inside the OM via `__ESBMC_alloca`
(`src/c2goto/library/python/list.c:52`), which ESBMC's memory model treats as a **non-expiring
dynamic object** (`goto-symex/dynamic_allocation.cpp`), and the Python variable holds a
**pointer** to it. Class instances differ only in being allocated as a frontend `code_declt`
in the caller's frame instead of through a non-expiring allocation.

**This is the whole fix in one sentence:** allocate class instances the way lists are
allocated (non-expiring dynamic object, variable holds a pointer), giving Python objects the
reference semantics CPython has.

---

## 3. Proposed design — reference semantics for instances

Migrate the instance representation from "value struct in a caller-frame `code_declt`" to
"pointer to a non-expiring dynamic `tag-Class` object":

1. **Allocation.** `ClassName(...)` allocates the object via a non-expiring primitive (an
   `__ESBMC_alloca`-backed `__ESBMC_new_object`-style intrinsic, or direct dynamic allocation),
   not a frontend `code_declt`. The constructor still runs with `self = <that pointer>`.
2. **Representation.** An instance-typed variable holds a **pointer** to `tag-Class`
   (it already is, for `None`/`Optional`-typed fields — `resolve_member_on_base` unwraps
   `pointer→struct`, so the read path is largely pointer-ready).
3. **Assignment is aliasing.** `b = a` copies the pointer (two names, one object) — matching
   Python and removing accidental value-copy semantics.
4. **Lifetime.** Heap objects never expire on return → the UAF disappears. No GC/free is
   modelled; leaking is sound for bounded verification (same as lists).
5. **`is None` / equality.** Null-pointer comparison is already the `None`/`Optional` model
   (memory `python-none-object-field-typing-4796`); pointer representation aligns the two and
   is expected to also resolve the `mk_eq` width-mismatch abort in #4796 (to be confirmed in
   Stage 2, not assumed).

### 3a. IR-level confirmation (2026-06-12)
GOTO dump of the failing `setup()` (branch `feat/python-object-heap-lifetime`):
```
DECL Node n1;  ASSIGN n1=NONDET(Node{long value; Node* next;});  FUNCTION_CALL Node(&n1, 1)
DECL Node n2;  ...                                               FUNCTION_CALL Node(&n2, 2)
ASSIGN n1.next = &n2;     // address of the stack-local n2
RETURN: n1               // struct returned by value; n1.next still points to expired n2
```
So the instance local is a **struct value** constructed in place via `&local`. `self` and the
`.next` field are *already* pointers, and the read path (`resolve_member_on_base`) already
dereferences pointer→struct — only the top-level local is a value. Confirmed change points:
`converter_stmt.cpp:3150-3185` (statement-call temp path) and the `get_var_assign` LHS path
(`:1890`+ / `handle_function_call_rhs :1152`) that emit `DECL Node x` + `Node(&x, …)`. Note a
redundant double construction (`Node(&$ctor_self$N,…)` *and* `Node(&local,…)`) to clean up too.

**Shortcut ruled out:** making the instance local *static-lifetime* (instead of stack) would
keep its address valid after return, but is **unsound** for objects constructed in a loop (all
iterations alias one static slot) — and `github_4831` (a baseline test) constructs in a loop.
Only per-construction non-expiring **dynamic** allocation (one fresh object per `ClassName(...)`
call, like the list OM) is sound.

### 3b. Implementation plan (concrete)
1. Add a non-expiring allocation for instances: at each `ClassName(...)`, emit a dynamic object
   (reuse the `__ESBMC_alloca`-backed mechanism the list OM uses, or a dedicated
   `__ESBMC_new_object(sizeof(tag-Class))` intrinsic) and bind the instance local to a
   **pointer** to it; call the constructor with that pointer as `self` (drop the extra `&`).
2. Make instance-typed locals/returns `pointer(tag-Class)`; assignment `b = a` becomes a pointer
   copy (alias). Audit `get_var_assign`, `handle_function_call_rhs`, return lowering, and the
   attribute read/write paths so a pointer base routes through the existing pointer→struct
   member access (largely already handled for `self`/fields).
3. Remove the redundant `$ctor_self$` double-construction.
4. Validate against the **26-test green baseline** + flip `github_4117_function_internal`; add a
   pass/fail regression pair for the return-escape case; dual-solver Bitwuzla+Z3 (Z3 in CI);
   Mode C on the constructor-lowering branch; `code-reviewer`.

**Scope reality:** this is a core representation migration across several deeply-coupled
converter functions with broad regression surface (26+ class tests) — a focused multi-cycle
effort, not a single edit. Branch `feat/python-object-heap-lifetime` holds the locked baseline
and this plan; the migration itself is the next dedicated unit of work.

---

## 4. Staged plan (each stage keeps the ≈40 passing class tests green)

- **Stage 0 — this design note + check-in.** ✅
- **Stage 1 — instance heap allocation + pointer representation.** Replace the constructor
  `code_declt`/`&temp` lowering with a non-expiring allocation returning a pointer; audit and
  unify attribute read/write and assignment to pointer semantics. **Acceptance:**
  `github_4117_function_internal` flips KNOWNBUG→CORE; all existing class tests
  (`inheritance`, `class*`, `github_4117_*`, `object_passing_*`, `self_ref_nested_attr_chain`,
  `github_4831*`, `github_4796_object_handle_eq`, …) stay green; new pass/fail regression pair
  for the return-escape case. Dual-solver; Mode C for the changed branch.
- **Stage 2 — escaping objects in containers / linked structures.** Target `reverse_linked_list`
  (#4796) and confirm whether the `mk_eq` width abort and the `__ESBMC_get_object_size`
  diagnostics on graph quixbugs resolve once objects are heap/reference. Flip what soundly flips;
  leave honest KNOWNBUGs where a separate blocker remains.
- **Stage 3 — #3067 `ClassHandler` refactor (separable).** Centralise the surface in §1a into one
  component with a real MRO (C3) and unit tests (inheritance, method lookup, static/class
  methods, multi-level bases). No behavioural change; pure maintainability + correctness for
  multi-level inheritance (currently single-hop).

### Sequencing trade-off (decision needed)
- **Lifetime-first (Stages 1→2→3):** fastest demonstrable KNOWNBUG flips, but Stage 1 edits the
  *scattered* code, so the allocation change touches several files.
- **Refactor-first (Stage 3→1→2):** the lifetime change then lands in one `ClassHandler`, but
  the refactor is a large no-functional-gain change up front and delays the first flip.

**Recommendation: lifetime-first.** It produces the first KNOWNBUG flip with the smallest
footprint and proves the reference-semantics model before investing in the refactor; #3067 then
consolidates code we already understand.

---

## 5. Risks & validation
- **Risk: value-vs-pointer assumptions.** Some class tests may implicitly rely on value copies.
  Mitigation: the ≈40 class tests are the regression net at every stage; bisect any flip.
- **Risk: double allocation / leaks inflating state.** Mitigation: one allocation per
  constructor call; verify unwind/memory budgets on the existing tests don't regress.
- **Risk: `--ir` mode interactions** (the OM width coupling of #4653 is orthogonal but shares
  `__ESBMC_alloca`); keep instance slots fixed-width.
- **Validation:** full Python regression (capped, subset by label), CPython sanity for any new
  test, dual-solver Bitwuzla+Z3 (Z3 leg in CI — this clone is Bitwuzla-only), Mode C for the
  constructor-lowering branch change.

## 6. Effort
Stage 1: medium (core lowering change + audit of attribute/assign paths). Stage 2: medium
(diagnosis-led, may surface follow-up blockers). Stage 3: large but mechanical (refactor + unit
tests). Stages are independently shippable PRs.
