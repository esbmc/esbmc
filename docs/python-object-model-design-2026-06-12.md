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

---

## 7. Stage 1 WIP progress (2026-06-12, branch `feat/python-object-heap-lifetime`)

First migration increment implemented (commits on this branch). **Status: IR-correct but does
NOT verify yet — do not merge.**

**What works (verified via `--goto-functions-only`).** Two edits — early pointer typing of the
LHS in `converter_stmt.cpp:get_var_assign` and heap-allocation of the instance in
`function_call/expr.cpp` constructor handling — now lower `n1 = Node(1)` to the intended
reference shape:
```
DECL Node * n1;
ASSIGN n1 = sideeffect cpp_new (Node)    // heap object, pointer
FUNCTION_CALL Node(n1, 1)                 // self = pointer (no &)
...
ASSIGN n1->next = n2;                     // pointer field write (NOT &n2 — no stack address)
RETURN: *n1                               // struct copy whose .next points to heap n2 (valid)
```
The UAF source (`&stack_local`) is gone; `n1->next = n2` stores a heap pointer.

**Blocker: the `cpp_new` sideeffect hangs symex.** `cpp_new` is normally lowered by the
clang-cpp **adjust** pass, which the Python pipeline does not run; the raw sideeffect reaches
symex with a malformed default initializer (`ASSIGN *n1 = nil`) and symex stalls immediately at
"Starting Bounded Model Checking" (no unwinding) even for the trivial `o = Node(1)` case. So
`cpp_new` is the wrong primitive here.

**Next step (clean primitive).** Allocate the instance the way the list OM does — via a C helper
that `malloc`s and returns the object pointer — instead of a raw frontend `cpp_new` sideeffect:
1. Add `void *__ESBMC_new_object(unsigned size) { return malloc(size); }` to the Python C OM
   library (`src/c2goto/library/python/`), rebuild (FLAIL + c2goto).
2. In `function_call/expr.cpp`, emit a call to it (cast `void*`→`Class*`) and use the result as
   `self`, replacing the `side_effect_exprt("cpp_new", …)`.
3. Then handle the remaining cascade: function return type for `-> Class` should become
   `pointer(tag-Class)` (currently `RETURN: *n1` copies the struct — works because inner pointers
   are heap, but a pointer return is cleaner); remove the redundant `$ctor_self$` double
   construction; audit attribute write / method-self / assignment-aliasing.
4. Validate against the 26-test class baseline + flip `github_4117_function_internal`.

---

## 8. Stage 1 WIP — malloc-helper works, flip achieved (2026-06-12)

The allocation primitive is solved and the target KNOWNBUG flips. **22/26 class baseline green +
`github_4117_function_internal` now SUCCESSFUL; 5 regressions remain (deeper cascade).**

**Working mechanism.**
- `__ESBMC_new_object(size)` added to `src/c2goto/library/python/list.c` — `malloc(size)` +
  `__ESBMC_assume(p != 0)` (CPython objects are never NULL; the constructor's `self` deref must
  be valid). Whitelisted in `symex_main.cpp` run_intrinsic dispatch (alongside
  `__ESBMC_list/dict/set`) so `bump_call` executes its body instead of the fatal
  "non-intrinsic prefixed with __ESBMC".
- `function_call/expr.cpp` ctor handling: `tmp = __ESBMC_new_object(sizeof(Class));
  current_lhs = (Class*)tmp; Class(current_lhs, …)`.
- `converter_stmt.cpp:get_var_assign` types the ctor LHS as `pointer(Class)` up front.
- Result: `o = Node(1); assert o.value==1` ✓; `def setup(): … return n1; assert n.next.value==2`
  ✓ (the flip); wrong-value variants ✗ as expected.
- **Build note:** editing `list.c` requires forcing the c2goto regen (`touch list.c` then
  rebuild) — plain incremental ninja missed it once, leaving the old `new_object` body (no
  assume) and a spurious NULL-deref. Confirmed when `libpython.c.o` rebuilds.

**5 regressions (cascade, next).**
- `class11`, `class13`, `class-attributes` — `array bounds violated`: the `malloc` size is
  `sizeof(Class)` at construction, but instance attributes added/shadowed later
  (`obj.class_attr = 2`) make the live struct larger than the allocated object. Need the
  allocation to cover the final struct size (or a typed allocation).
- `object_passing_comprehensive` — `assert v1.x == 10` fails: instance value lost across
  function-argument passing; pointer-aliasing semantics for instance args need auditing.
- `github_4796_object_handle_eq` — `__eq__` / linked-list path (no verdict / loop).

These are exactly the §4-Stage-1 "audit attribute write / method-self / assignment-aliasing"
items. The allocation foundation is done; the remaining work is making the rest of the converter
consistently pointer-aware.

---

## 9. Stage 1 WIP — cascade mostly cleared, 26/27 (2026-06-12)

Switched from byte-`malloc` to a **typed object allocation** intercepted in symex, and cleared
4 of the 5 cascade regressions. **26/27 class baseline + `github_4117_function_internal` flips.**

**Allocation (final form).** `o = ClassName(...)` → `o = __ESBMC_new_object()` (a no-arg stub in
`list.c`); symex (`symex_main.cpp` run_intrinsic) intercepts it and allocates a typed,
non-expiring object of the class struct **carried by the LHS pointer type** via `symex_mem_inf`
— the same mechanism `__ESBMC_create_inf_obj` uses for PyObj. This is sized *symbolically* by
the struct type, so it is robust to the struct still gaining fields after the construction site
(which a byte-`malloc` of `sizeof` cannot handle — the struct grows when usage adds/​shadows
instance attributes, e.g. `obj.class_attr = 2`; that was the `class11`/`class13`/`class-attributes`
`array bounds violated`). Dead ends ruled out: byte-`malloc` (wrong size on lazy struct growth),
raw `cpp_new` sideeffect (hangs goto-gen — no clang-cpp adjust), `alloca` (frame-freed),
finite single `dynamic_*_value` via `create_dynamic_memory_symbol` (binding came out broken —
all reads failed; left for a follow-up, it is the right perf fix).

**Aliasing.** Extended the early pointer-typing in `get_var_assign` to use `flow_rhs_class`, so
`b = a` (alias of an instance) types `b` as the same class pointer — a pointer copy (shared
object), not a struct copy. Fixes `github_4117_alias_chain` (was `n1->next = &alias` with `alias`
already a pointer).

**Remaining: `github_4796_object_handle_eq` (perf regression).** Was CORE; now no verdict inside
the cap. The infinite-array object (`symex_mem_inf`) makes the `is`/`==` identity reasoning over
a linked-list `reverse` blow up in the solver. The fix is a **finite** single typed object
(symbolically sized but not an infinite array); the `create_dynamic_memory_symbol` attempt did
not bind correctly and needs debugging. Until then this CORE test regresses, so the branch is
**not mergeable yet**.

---

## 10. Stage 1 — finite typed object + github_4796 root cause (2026-06-12)

Replaced the infinite-array allocation with a **single typed `dynamic_*_value`** object (mirrors
`symex_mem`'s `size_is_one` path: struct-typed symbol, `address_of2tc(struct_type, sym)` →
`struct*` then cast to the LHS pointer, 3-arg `track_new_pointer`, `auto_deallocd=false`). Key
bug fixed along the way: `address_of2tc`'s first argument is the **pointee** type, so passing the
LHS pointer type produced a `Class**` double pointer and broke every read. The single value is
cleaner and more efficient than the infinite array and avoids its identity-reasoning cost.

**Result: 26/26 other class tests pass + `github_4117_function_internal` flips.** No regressions
except `github_4796_object_handle_eq`.

**`github_4796` root cause (separate from allocation).** `reverse()` returns `prevnode`, which is
`None`-initialised then reassigned the `Node` parameter inside the loop. Return-type inference
keeps it as the scalar **None-handle** (`unsigned long int`), not `Node*`. So `r = reverse(a)` is
a scalar handle while `b = Node(2)` is a pointer, and `r == b` / `r != b` is the `#4796`
*handle-vs-value* reconciliation — which, under the new pointer instances, blows up the SMT
encoding (it stalls in "Encoding remaining VCC(s)"; isolated to `test_distinct`'s distinct-object
comparison). Minimal same-object identity (`r is n`), the reverse loop, the `None`-default
signature, and the imported class all verify fine **individually** — only the
returns-a-None-initialised-instance + cross-object `==`/`!=` combination triggers it.

The fix belongs to the None/Optional model (#4796/#4653): a variable that is `None`-initialised
then assigned a class instance must be typed as the instance **pointer** (`Optional[Class]` ⇒
`Class*` with NULL for None), so both sides of `==` are class pointers and the comparison is
clean pointer identity. Note `flow_rhs_class`-based aliasing only fires at module depth-1, so it
does not retype `prevnode` inside `reverse()`. **Branch remains not mergeable until this lands.**

---

## 11. github_4796 fixed via comparison reconciliation — but broader blast radius found (2026-06-12)

**`github_4796` fixed (27/27 class baseline).** Root cause was *not* the allocation: the `#4796`
handle-vs-value `==`/`is` reconciliation in `converter_binop.cpp` only matched a **by-value
struct** instance, but the migration makes instances **`Class*` pointers**, so the reconciliation
did not fire and `handle == Class*` fell through to a blow-up encoding. Extended it to also accept
a **class pointer** side (cast the None-handle to that pointer type; skip `gen_address_of` since
the side is already a pointer). With that, the whole 27-test class baseline passes **and**
`github_4117_function_internal` flips.

**Broader blast radius (NOT mergeable).** A wider risk-subset run (comparison / None / Optional /
dunder / dataclass) surfaced **≥3 regressions the class baseline did not cover**:
- `github_3976_optional_attr_access` — `box: Optional[Box] = None; box = Box(7); box.value`.
  GOTO shows `box` typed `unsigned long int` (the Optional **handle**), but the migration
  constructs into it as `Box*` (`box = new_object(); Box(box, 7); box->value`) → out-of-bounds.
- `dataclass-edge-equality_true`, `dunder-bool-condition` — equality / dunder paths that assume
  by-value struct instances.

Root: `NoneType`/`Optional` are modelled as pointer-width **`unsignedbv` handles**
(`type_handler.cpp:457-464`) and `Optional[X]` as a `tag-Optional_` struct — neither aligned with
pointer instances. **Stage 1 (object lifetime) and Stage 2 (None/Optional → `Class*` unification)
are therefore entangled**: `Optional[Class]` and `None`-initialised instance variables must become
`Class*` (NULL for None) for the migration to be sound across the suite. That unification is the
required next step before this branch can merge; the class baseline alone understated the scope.

---

## 12. None/Optional unification — attempted, scoped as multi-layer (2026-06-12)

Took on the None/Optional → `Class*` unification needed to clear the broader regressions
(github_3976 etc.). Made the **typing layer** work but found the unification is multi-layer and
larger than one change; reverted the partial attempt to keep the 27/27 + `github_4796` state clean.

**Typing layer (prototyped, works).** Extended the `get_var_assign` early pointer-typing to treat
a target annotated `Class` / `Optional[Class]` / `Class | None` as `Class*`. GOTO confirms it:
`box: Optional[Box] = None` now lowers to `DECL Box * box; box = 0` (NULL), and `box = Box(7)`
allocates and constructs through the pointer. So the **declaration/None side** unifies cleanly.

**But field access + value representation do NOT yet unify (the deeper layers).** With `box: Box*`,
`box.value == 7` still lowers as an *object* comparison —
`(__ESBMC_PyObj *)box->value == (void *)7` — and the in-`__init__` `self.value = value` goes
out of bounds. Reasons:
- Field access on an `Optional`-origin variable routes `x.value` through the None/object-handle
  path (treating the field as a `PyObj` handle) rather than the concrete `Box.value : int` field.
- The `==` lowering then takes the object-identity path for `x.value`, not integer comparison.

So the unification needs, beyond typing: (a) field-read/-write on a `Class*` to resolve the
concrete struct field regardless of an `Optional` origin, and (b) the comparison/`bool`/dunder
paths to treat a `Class*` field value by its declared type. That is the `#4653`/`#4796` object/
Optional model rework — a multi-layer change across `type_handler` (Optional⇒`Class*`+NULL),
attribute access (`converter_expr`), and value/dunder lowering (`converter_binop`,
`converter_dunder`). It is the correct next unit of work but is larger than a single pass.

**State:** branch holds Stage 1 (object lifetime) + the `#4796` class-pointer comparison fix —
27/27 class baseline, `github_4117_function_internal` flips. It is **not mergeable** until the
None/Optional unification lands (≥3 known broader regressions: `github_3976_optional_attr_access`,
`dataclass-edge-equality_true`, `dunder-bool-condition`).

---

## 13. None/Optional unification — confirmed deeply intertwined, beyond a single pass (2026-06-12)

Implemented and tested the unification across layers; confirmed it is a **deep, intertwined model
rework** that cannot be landed soundly in isolation, and reverted to keep the clean state.

Findings:
- **Typing layer alone regresses the flip.** Typing annotated `Class`/`Optional[Class]` targets as
  `Class*` (broadening the early-typing gate) **breaks `github_4117_function_internal`** (the very
  flip this work achieved) — the broadened gate/typing perturbs non-Optional class variables too.
  So the typing layer cannot be applied independently of the rest.
- **Field access + construction under Optional are separately broken.** With `x: Optional[Box]`
  typed `Box*`: the in-`__init__` `self.value = value` goes out of bounds, and `x.value == 7`
  lowers as an object comparison `(__ESBMC_PyObj *)x->value == (void *)7`. The same program without
  the `Optional` annotation (`x = Box(7)` / `x: Box = Box(7)`) verifies cleanly
  (`ASSERT x->value == 7`). So the `Optional` annotation specifically contaminates the struct
  build and the comparison path (via `tag-Optional_` / the `PyObj` handle model).

Conclusion: a sound None/Optional → `Class*` unification must be co-designed across
`type_handler` (Optional ⇒ `Class*`+NULL, without a parallel `tag-Optional_` struct), the early
pointer-typing (`get_var_assign`, without perturbing non-Optional class vars or the flip),
attribute access (`converter_expr`), and value/comparison/dunder lowering (`converter_binop`,
`converter_dunder`) — changing any one in isolation regresses either the flip or the Optional
cases. It is a genuine multi-pass `#4653`/`#4796` model rework and the correct next dedicated unit
of work.

**Final branch state:** Stage 1 object lifetime + the `#4796` class-pointer comparison fix —
**27/27 class baseline, `github_4117_function_internal` flips, `github_4796` passes**. Not
mergeable: ≥3 broader regressions (`github_3976_optional_attr_access`,
`dataclass-edge-equality_true`, `dunder-bool-condition`) await the None/Optional unification.
