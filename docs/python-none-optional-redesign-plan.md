# Python None/Optional → `Class*` Model Redesign — Project Plan (2026-06-12)

**Status:** Planning. Scoped as a standalone project, to be executed in its own focused effort
rather than incrementally on top of the object-lifetime branch.

**Companion:** `python-object-model-design-2026-06-12.md` (the Stage-1 object-lifetime migration
and §§11–14, which record the empirical findings this plan is built on). **Prerequisite:** the
object-lifetime migration (instances as non-expiring `Class*` heap objects) on branch
`feat/python-object-heap-lifetime` (`#3067`/`#4773`).

---

## 1. Goal

Unify the representation of `None` and `Optional[Class]` (for **user classes**) with the
object-lifetime migration's pointer instances, so that:

- a variable typed `Optional[Class]` / `Class | None` is a `Class*` whose NULL value is `None`;
- field access, equality (`==`/`!=`), identity (`is`/`is not`), truthiness (`if x:`), and dunder
  dispatch on such a variable all operate on the concrete class by its declared field types;

and thereby clear the three regressions the migration introduced
(`github_3976_optional_attr_access`, `dataclass-edge-equality_true`, `dunder-bool-condition`)
while keeping the 27-test class baseline green, the `github_4117_function_internal` flip, and
`github_4796_object_handle_eq`.

**Scope boundary.** This project covers `Optional[<user class>]` and `<user class> | None` only.
`Optional[int]`, `Optional[str]`, etc. (optional *scalars/builtins*) are **out of scope** — those
are not pointers and keep the existing handle / `tag-Optional_` representation. Narrowing to user
classes is what makes the project tractable.

---

## 2. Current model (precise)

- `NoneType` and bare `Optional` lower to a **pointer-width `unsignedbv` handle**
  (`src/python-frontend/type_handler.cpp:454-464`). `None` is the integer `0`.
- `Optional[X]` builds a wrapper struct `tag-Optional_<X>` (`type_handler.cpp:1383`).
- After the object-lifetime migration, **instances are `Class*` pointers** allocated via
  `__ESBMC_new_object` (intercepted in `goto-symex/symex_main.cpp` → a single typed dynamic
  object) and constructed in place (`function_call/expr.cpp` constructor handling).
- `converter_expr.cpp:453-482` already unwraps `tag-Optional_*` and pointer-to-struct on
  attribute access.
- The `#4796` `==`/`is` reconciliation (`converter_binop.cpp:587`) was extended (already on the
  branch) to compare a **class-pointer** side against a None-handle as pointer identity.

**The mismatch.** A None-able class variable is a scalar handle; an instance is a `Class*`
pointer. The two reach the solver / lowering with incompatible shapes, producing: the `mk_eq`
width abort (`#4796`, worked around for the class-ptr-vs-handle compare), **field-value boxing**
(`self.value = 7` stored as a struct `{.value=7}` → out-of-bounds), and object-routed comparisons
(`(__ESBMC_PyObj *)x->value == (void *)7`).

---

## 3. Why incremental patching failed (lessons from 6 attempts)

Recorded so the redesign does not repeat them:

1. **Broad early-typing regresses the flip.** Typing *every* annotated class Name as `Class*`
   perturbs non-Optional class variables and breaks `github_4117_function_internal`.
2. **Narrow (Optional-only) early-typing is safe for locals** but does not reach **module
   globals**, which are *preregistered* (`preregister_global_variables`,
   `converter_stmt.cpp:1369`) with their handle type before `get_var_assign` runs.
3. **Typing a var as `pointer(tag-Class)` before the class struct is built segfaults.** A
   pointer to a not-yet-`ensure_sym`'d class symbol reaches `migrate_type_back(null)` via the
   `new_object` symex interception (`to_pointer_type(ret).subtype` is null) — and, crucially,
   this corrupted **unrelated** programs (`fn.py`, which has no `Optional`, crashed). The class
   struct symbol must exist and be resolvable before any pointer-to-class typing is used for
   allocation.
4. **Field-value boxing is a separate, deeper defect.** Even with the variable correctly typed
   `Class*`, the presence of an `Optional[Class]` variable contaminates the class field's
   inferred type into a boxed struct; the same class without an `Optional` variable keeps
   `value : int` and verifies. The boxing lives in the field/attribute-type inference, not the
   variable typing.

**Conclusion:** the sub-problems are interdependent through (a) the class-build ordering and
(b) shared field/value-type inference. They cannot be landed one patch at a time; the
representation change must be coherent across all touch-points and sequenced build-first.

---

## 4. Target model

For a user class `C`:

- `Optional[C]` and `C | None` annotations ⇒ `pointer(tag-C)` (`C*`). No `tag-Optional_C` wrapper.
- `None` assigned to a `C*` lvalue ⇒ the NULL pointer (`gen_zero(pointer)` / a typed NULL).
- `if x:` / truthiness on `C*` ⇒ `x != NULL`.
- `x.attr` on `C*` ⇒ the concrete `tag-C` member by its declared type (existing
  pointer→struct member access; no boxing).
- `a == b` / `a != b` / `a is b` / `a is not b` where either side is `C*` and the other is
  `C*` or `None` ⇒ pointer identity (the `#4796` class-pointer path, already on the branch).
- A field of `C` declared/assigned a scalar (`self.value = value`, `value: int`) keeps the
  scalar field type regardless of any `Optional[C]` variable elsewhere.

`None` where the static type is genuinely unknown (no `Optional[C]` context) keeps the existing
pointer-width handle — this project does not remove the handle globally, only for `Optional[C]`.

---

## 5. Subsystem change inventory

Each item lists the file, current behaviour, and the required change. They must be designed
together; the **build-ordering** item (D) gates the typing items (A/B).

- **A. Annotation typing — `type_handler`.** Map `Optional[<user class>]` / `<class> | None` to
  `pointer(tag-class)` in the annotation→type resolution (`type_handler.cpp:237` returns the
  outer `"Optional"` today; it must yield the inner class as a pointer for user classes). This is
  the single source of truth so locals, globals, params, returns, and fields all agree.
- **B. Variable typing sites consuming A.**
  - `get_var_assign` early-typing (`converter_stmt.cpp` ~1530): locals.
  - `preregister_global_variables` (`converter_stmt.cpp:1369`): module globals — must use A, not
    the legacy `extract_type_info` handle.
  - Function parameters and return types annotated `Optional[C]` (so e.g. a `def f(x: Optional[Node])`
    param and an `-> Optional[Node]` return are `Node*`). This is what would also let a generic
    function returning a `None`-initialised-then-instance variable type as `Node*` (the
    `github_4796` shape) — though that case is already handled by the comparison fix.
- **C. `None` literal lowering.** `x = None` / `x: Optional[C] = None` where the lvalue is `C*`
  ⇒ NULL pointer of the right type (today `None` is the integer `0`; ensure it becomes a typed
  NULL so pointer ops and `is None` work).
- **D. Class-build ordering (the segfault gate).** Before any variable/field is typed
  `pointer(tag-C)`, `tag-C` must be registered (`ensure_sym`, `python_class_builder.cpp`) and
  resolvable so `new_object`'s `migrate_type_back`/`ns.follow` cannot hit a null/unbuilt symbol.
  Options: (i) eagerly `ensure_sym` the class when an `Optional[C]` annotation is first seen;
  (ii) make the `new_object` interception robust to an unresolved subtype (defensive, but masks
  the real ordering bug); (iii) hoist class discovery/registration into a pre-pass over the
  module body before any typing. **(iii) is preferred** — a deterministic pre-pass that
  `ensure_sym`s every user class up front removes the ordering hazard for the whole project.
- **E. Field / attribute-type inference — kill the boxing.** Ensure a class field declared/assigned
  a scalar keeps its scalar type even when an `Optional[C]` variable exists; remove the path that
  boxes `self.value = 7` into a struct. Locate via the counterexample shape
  `self->value = { .value = 7 }` (the field is being widened to an object/struct). Likely in the
  attribute/field type inference under the `Optional`/`tag-Optional_` model.
- **F. Comparison / truthiness / dunder lowering.** With A–E in place, `==`/`is`/`bool`/dunder on
  `C*` should already route to pointer identity / concrete-type ops (the `#4796` class-pointer
  reconciliation is on the branch). Audit `converter_binop.cpp` and `converter_dunder.cpp` for any
  remaining `is_object_handle`/`tag-Optional_` assumptions that need the `C*` case.
- **G. Remove / bypass `tag-Optional_C`.** For user classes, the `tag-Optional_<C>` wrapper
  (`type_handler.cpp:1383`) should no longer be produced; audit its readers (`converter_expr.cpp`
  Optional unwrap) so they tolerate its absence for user classes (keep it for `Optional[scalar]`).

---

## 6. Sequencing (build-first, representation-coherent)

Land in this order, each step validated by GOTO inspection (not just test pass, since tests only
go green once the chain is complete):

1. **D first** — class pre-registration pass; verify no `pointer(tag-C)` typing can reach an
   unbuilt symbol (re-run the full suite to prove no new crashes, *before* any typing change).
2. **A + G** — annotation resolution yields `C*` for `Optional[user class]`, and stop emitting
   `tag-Optional_C` for user classes; keep scalar-Optional behaviour.
3. **B + C** — wire A into local/global/param/return typing and make `None` a typed NULL. Verify
   the flip and the 27 class baseline stay green at every sub-step (the flip is the canary).
4. **E** — remove field-value boxing; verify `self.value = 7` lowers to a scalar store and
   `github_3976` construction is in bounds.
5. **F** — audit/clean comparison/dunder; verify the 3 regressions pass.
6. Full regression sweep + the broader risk subset.

---

## 7. Risks & mitigations

- **Non-obvious coupling / crashes** (seen: `fn.py` crashing from an unrelated typing change).
  *Mitigation:* land **D** first and re-run the full suite for crashes before touching typing;
  treat the flip + 27 baseline as canaries after every sub-step; bisect any crash immediately.
- **Scalar-Optional regressions.** Touching the Optional path risks `Optional[int]`/`Optional[str]`.
  *Mitigation:* gate every change on `is_class(inner)`; add explicit `Optional[int]`/`[str]`
  regression coverage to the canary set.
- **Field-boxing fix over-reaches.** *Mitigation:* characterise the boxing precisely (which
  inference path widens the field) before editing; add a focused test (`self.value:int` under an
  `Optional[C]` variable).
- **Dataclass / dunder have independent roots.** `dataclass-edge-equality_true` and
  `dunder-bool-condition` may need targeted work beyond A–G; scope them as separate verification
  items, not assumed-fixed by the typing unification.

---

## 8. Validation & acceptance

**Must stay green throughout (canaries):** `github_4117_function_internal` (flip),
`github_4796_object_handle_eq`, the 27-test class baseline, and explicit `Optional[int]`/
`Optional[str]` cases.

**Acceptance:** the 3 regressions pass — `github_3976_optional_attr_access`,
`dataclass-edge-equality_true`, `dunder-bool-condition` — plus a new regression pair for the
core contract (a local *and* a global `Optional[C]` constructed, field-accessed, compared `==`/
`is`/`!=`, and truthiness-tested), CPython-valid for `check_python_tests.sh`. Then a full
`ctest -L python` sweep with zero new failures (cap/narrow per the repo's 5-minute rule), and the
broader comparison/None/Optional/dunder/dataclass risk subset. Dual-solver where the affected
tests pin solvers; this clone is Bitwuzla-only, so the Z3 leg runs in CI.

---

## 9. Effort

Multi-day, multi-pass. Milestones map to §6 steps 1–6; step 1 (D, the build pre-pass) is the
highest-leverage de-risking and should be done and proven crash-free in isolation first. Only
after D is solid should the representation/typing changes (A/B/C/G) land together, followed by the
boxing fix (E) and the comparison/dunder audit (F).

---

## 10. Step-D execution finding (2026-06-12) — the incomplete global pre-pass is wrong

Executed option (i)/(iii) of §5-D: a `preregister_classes` pass that, before any conversion,
calls a new `python_class_builder::ensure_symbol()` for every `ClassDef` in the module (and
imported modules), creating the incomplete `tag-<class>` struct symbol up front.

**Result: NOT behaviour-neutral.** Canaries (`fn.py` flip, `simple.py`, `github_4796`) stayed
green, but `object_passing_comprehensive` (44 classes) regressed to
`ERROR: _init_undefined / migrate expr failed` during *Generating GOTO Program*. Root cause: a
variable of a class type that is registered **incomplete** but not yet fully built is declared
with `gen_zero` over an incomplete struct, yielding the `_init_undefined` placeholder that fails
migration. Pre-registering *incomplete* symbols for classes that are referenced before (or
without) their lazy completion is unsafe. Reverted; tree restored.

**Refined design for step D — build-on-demand at the typing site, not a global pre-pass.**
The de-risking the typing step needs is: *when a pointer-to-class type is about to be created for
a variable* (the `Optional[Class]` typing in step B), **build that specific class first** (the
existing `process_forward_reference` / `get_class_definition` path, which produces a **complete**
struct), then form `pointer(tag-Class)`. This guarantees the tag symbol exists and is complete at
the point of use, without registering incomplete stubs for unrelated/unbuilt classes. Concretely:

- Drop the standalone `preregister_classes` pass.
- In each typing site that yields `pointer(tag-C)` (locals in `get_var_assign`, globals in
  `preregister_global_variables`, params/returns), first ensure `C` is built on demand. For the
  **global preregister** — which runs before the main class-build loop — this means triggering
  `C`'s build there (it currently only registers variable symbols), so a global typed `C*` never
  references an unbuilt/incomplete `C`.
- Keep `option (ii)` (defensive null-subtype guard in the `new_object` symex interception) as a
  belt-and-braces safety net, but the real fix is the on-demand complete build above.

So step D folds into step B rather than preceding it as a separate pass. The sequencing in §6
updates to: **(1) on-demand complete-build helper at the typing sites → prove no crash with the
typing of a single Optional[Class] local *and* global; (2) A+G; (3) the rest as before.** The
"prove crash-free before broad typing" discipline still holds — validate the build-on-demand on
one local and one global Optional[Class] (no `_init_undefined`, no segfault) before wiring it
everywhere.

---

## 11. Step-B execution (2026-06-12) — build-on-demand typing landed, crash-free + a key discovery

Implemented the build-on-demand `Class*` typing for `Optional[Class]` / `Class | None` (commit
`873ef78ad4`):

- `annotated_optional_class()` extracts the user class from `Optional[C]` / `C | None`.
- **Local** (`get_var_assign`): on such an annotation, `process_forward_reference` builds the
  class on demand (complete struct), then type the LHS `Class*`.
- **Global** (`preregister_global_variables`): type the global `Class*` (a zeroable NULL
  pointer; the main class-build loop completes the struct — the global's own NULL value needs no
  complete struct, so no `_init_undefined`, the failure mode of the incomplete pre-pass in §10).

**Proven crash-free + non-regressive (clean binary):** 0 crashes across the class baseline +
Optional/None subset; the local case (`github_3976` `box`) and the **global** case (`a.py` `x`)
both lower to `Box*` and reach a verdict (was a segfault for the global). 23/23 class baseline
unchanged; 39/40 Optional/None pass. The 3 target regressions still FAIL on field-value boxing
(step E, next), as expected — typing alone does not de-box.

**Critical discovery — the session's "mysterious crashes" were build artifacts.** Every prior
attempt saw *unrelated* programs (`fn.py`, `inheritance`) hang/crash after a change, which led to
the wrong conclusion that the typing was fundamentally fragile. Root cause: **incremental builds
after a `python_converter.h` change produce ABI-inconsistent objects** (some TUs compiled against
the new header, others stale), which corrupts execution of programs that don't touch the feature.
A **clean python-frontend rebuild** (delete `build/**/python-frontend/**/*.o`, rebuild) makes the
hangs/crashes vanish. **Process rule for the rest of this project: after any change to
`python_converter.h` (or other widely-included headers), force a clean frontend rebuild before
judging behaviour.** This very likely also explains the earlier "segfault on typing globals."

**Next:** step E (field-value boxing removal) — make a class field declared/assigned a scalar
keep its scalar type under an `Optional[C]` origin, so `self.value = 7` stores an int (not
`{.value=7}`) and `x.value == 7` is an integer compare. Then F (comparison/dunder audit).

---

## 12. Step-E execution finding (2026-06-12) — it is NOT field-type boxing; it's the allocation

Implemented step B (build-on-demand typing) and went to fix the "field-value boxing". The boxing
turned out to be a symptom of a deeper issue in the **object allocation for Optional-typed
instance variables**, not the field *type*:

- Counterexample for `x: Optional[Box] = None; x = Box(7)` (`d.py`): `self = (Box *)(&dynamic_1_value)`
  with `alloc_size = 1` — the `__ESBMC_new_object` symex (`symex_main.cpp`) allocated a **too-small
  dynamic object** (it needs a cast to `Box*`), so `self->value = value` is out of bounds.
- The non-Optional global `x: Box = Box(7)` (`c.py`) does **not** use `new_object` — its global
  keeps the struct type and constructs in place (`Box(&x, 7)`), which works. Only the
  Optional-typed (pointer) variable takes the `new_object` path.
- Paradox: a ctor-typed local (`n1 = Node(1)`, `fn.py`) and an Optional-typed local
  (`box: Optional[Box]`, `github_3976`) end up with the **identical** `pointer(tag-class)` type
  (both via `get_typet` → `symbol_typet("tag-...")`), yet `n1`'s `new_object` allocates a complete
  `Node` and `box`'s allocates a too-small `Box`. Confirmed real (survives a clean rebuild, so not
  the build-artifact issue from §11).

**So step E is mis-named** — there is no field-type widening to undo. The defect is that
`ns.follow(migrate_type_back(base))` in the `new_object` interception yields an incomplete/too-small
struct for the Optional-typed class, even though the class is built (the redundant `$ctor_self$`
DECL shows the complete `Box` struct exists). Either `base` (the LHS pointer's subtype) is not the
`tag-class` symbol expected, or `ns` does not resolve it to the completed struct at that point.

**Actionable next step (needs interactive symex debug, not static analysis):** at the
`has_prefix(symname, "c:@F@__ESBMC_new_object")` block in `symex_main.cpp`, log `base`,
`migrate_type_back(base)`, `ns.follow(...)`, and the `tag-<class>` symbol's completeness, for `d.py`.
Compare against the working `fn.py` `n1` case. The fix is then either (a) resolve `base` to the
completed `tag-class` (build-ordering), or (b) make the new_object allocation robust to a
symbol-typed subtype. Step B (typing) is correct and committed (`873ef78ad4`); this allocation fix
is the true content of the next unit.

---

## 13. Step C landed (2026-06-12) — None-on-Class* + dunder dispatch through pointers

The §12 "too-small allocation" was a symptom, and the actual root cause was upstream in the
**frontend typing**, not the symex allocation. `none_type()` is `pointer_typet(bool_typet())`
(`src/util/python_types.cpp`), i.e. **pointer-to-bool**. In `handle_assignment_type_adjustments`
(`converter_stmt.cpp`), the `rhs.type() == none_type()` branch *unconditionally* retyped the
lvalue symbol to `none_type()`. So `x: Optional[Box] = None` overwrote `x`'s freshly-computed
`Box*` type (from step B) with `pointer(bool)`. The later `x = Box(7)` then drove `new_object`
with `base = bool`, allocating a bool-sized object — exactly the §12 counterexample. Proof:
`x: Optional[Box]` **without** `= None` allocated a correct `Box` (`base = symbol → struct`); adding
`= None` flipped `base` to `bool`.

**Step C fix.** In that branch, if the lvalue is already a user-class reference
(`is_user_class_pointer`), keep its `Class*` type and assign a `typecast(None, Class*)` typed NULL
instead of retyping to pointer-bool. Non-class targets keep the legacy retype.

**Two further migration regressions, same root family.** With instances now `Class*` pointers,
two dispatch sites still assumed by-value structs:
- `dispatch_dunder_operator` (`converter_dunder.cpp`) checked `lhs_type.is_struct()` and passed
  `gen_address_of(operand)`. For a `Class*` operand it skipped dispatch (no struct) and, when it
  did fire, produced `Class**`. Fix: `resolve_operand_type` follows a `Class*` to its pointee
  struct; a new `dunder_ref_arg` passes the pointer through (a by-value struct still gets its
  address taken). Restored dataclass `a == b` (synthesised `__eq__`).
- The `__bool__` truthiness site (`get_conditional_stm`) passed `gen_address_of(bool_object)`,
  yielding `Class**` for a migrated instance. Fix: pass the pointer directly when it is one.
  Restored `if obj:` (`dunder-bool-condition`).

**Hardening.** Both new paths gate on `is_user_class_pointer` — a pointer whose pointee struct tag
resolves to a real user class via `json_utils::is_class`. `is_excluded_struct_tag` additionally
excludes the `tag-struct __ESBMC_Py...` model structs (list/object/slice), which under the
migration are *also* pointer-to-struct but own their own operator paths. Without this, a `list`
lvalue/operand would wrongly enter the class-pointer branches (caught in code review).

**Status:** committed `11f74327c9`. Three regressions cleared: `github_3976_optional_attr_access`,
`dataclass-edge-equality_true`, `dunder-bool-condition` (all FAIL at the pre-branch baseline, PASS
now; they are the Phase-2 contract regressions). `dunder-bool-condition-fail` and the `github_4796`
handle-eq canaries hold. A class/dunder/dataclass/object/optional family sweep shows **zero new
failures** vs. the step-B baseline; the remaining family failures (threading_*, `inheritance*`,
`class-attributes-scoped`) are **pre-existing** migration breakage (verified by re-running them at
the baseline binary), not introduced here.

**Next (still open in the migration):** the class-typed-**parameter** path — e.g.
`def f(obj: MyClass) -> MyClass` (`class-attributes-scoped`, `inheritance2`) — still hangs/fails
under the pointer model. That is the next scoped unit (param/return typing + super().__init__
chains), independent of the None/Optional typing now closed by step C.

---

## 14. Class-typed parameters and class globals landed (2026-06-12)

The "class-typed parameter/return path that still hangs" was three distinct
defects, all now fixed (commit `9abe26eacc`):

1. **Class-typed parameter** (`def f(obj: C)`) was typed as a by-value struct
   `C`, but the migrated caller passes a `C*`. The mismatch built a malformed
   call expr; `goto_symex_statet::rename_type` then dereferenced a null type2tc
   (`is_array_type`, `EXC_BAD_ACCESS` at offset 0xc). Fix: type the formal `C*`
   (like `self`) in `register_function_argument`. The existing struct→pointer
   argument coercion (`converter_funcall.cpp` ~1061) handles a by-value struct
   argument by taking its address, so both pointer and by-value call sites work.
   This also gives correct Python reference semantics — a callee mutating the
   parameter is visible to the caller.

2. **Class return** (`-> C`) already worked (instances are `C*`).

3. **Annotated module-global class instance** (`m: C = C()`) crashed even with
   no parameter or attribute access. Root cause was upstream in
   `preregister_global_variables`: `extract_type_info` returns a
   *default-constructed* `typet` (id `""`, which is **neither** `nil` **nor**
   `empty` — those are ids `"nil"`/`"empty"`) when the annotation is not yet in
   its final `id`-bearing form at pre-registration time. The old skip guard
   (`is_nil() || is_empty()`) missed the empty-id placeholder, so `m` was
   registered with an invalid type. `move_symbol_to_context` does **not**
   overwrite a plain variable, so the later, correctly-typed creation in
   `get_var_assign` was discarded and `m` kept the empty type — the constructor
   then built `&m` over an empty-typed lvalue and crashed. Fix: also skip when
   `var_type.id().empty()`, leaving `m` for `get_var_assign` to type as `C*`;
   and register an annotated plain-class global as `C*` when it does resolve.

Shared helper `is_user_class_struct_type` reads the class tag from a struct or
an unresolved `tag-<Class>` symbol (no `ns.follow`, no build dependency) and
validates via `json_utils::is_class`, which naturally excludes the
`tag-struct __ESBMC_Py...` model structs. `is_user_class_pointer` is now a thin
wrapper over it.

**Cleared:** class-attributes-scoped, inheritance, github_4543_is_none /
is_not_none, object_passing_comprehensive_fail. New tests:
`object_param_global{,_fail}`. Family sweep: zero new failures.

**Two newly-isolated, orthogonal pre-existing bugs (NOT this path; still open):**
- `sound: int = obj.method()` where the method returns a **str** (or any type
  mismatching the variable annotation) lowers the call to
  `ASSIGN sound = NONDET(int)` — the **call is dropped entirely**, so its side
  effects (e.g. `self.energy -= 5`) never run. Sole remaining cause of
  `inheritance2` (`assert dog.energy == 85` fails because `bark()` was elided).
  Independent of inheritance and of the object model.
- `c.value += by` — an **augmented** assignment to an attribute through a
  `Class*` parameter — crashes in Converting (plain `c.value = v` is fine).

The next migration unit is the Thread-subclass / module-scope construction
cases (`threading_thread_subclass_*`), still failing on the branch.

---

## 15. Call-drop on result type-mismatch — FIXED (2026-06-12)

One of the two orthogonal bugs noted in §14 is fixed (commit `e3f9abb040`).

`n: int = obj.method()` where the method returns a **str**: the assignment is
skipped (a char* cannot be stored in an int slot — the "Python dynamic typing"
skip path in `get_var_assign`). That path *intended* to still emit the call as
a void call so side effects survive, but only matched the
`side_effect_function_call` **expression** form. A method call arrives as an
already-lowered `code_function_call` **statement** (nil result), fell through,
and was dropped entirely — so `self.energy -= 5` inside the method never ran.
Fix: handle the `code_function_call` form too (emit `void_call` from its
function + arguments). The call now runs; only the mismatched result is dropped.
Tests: `method_sideeffect_str_into_int{,_fail}`.

**inheritance2 is still red**, now for the *other* orthogonal reason: line 30
`assert sound == "Woof!"` where `sound: int = dog.bark()`. CPython ignores the
`int` annotation at runtime and `sound` holds the str, so the assert passes;
ESBMC keeps `sound` an int slot (the str result is dropped), so it fails. This
is the **dynamic-retype-inside-a-function** limitation: the "straight-line
dynamic retyping" path that rebinds `n` to str is gated to `block_nesting_ == 1`
(module top level) because conditional retyping at a control-flow join is
unsound (#4770/#4774). Confirmed general, not call-specific: `n: int = "hello"`
also fails inside a function but passes at module scope. Lifting the gate
soundly (distinguishing an unconditional function-body statement from one inside
an `if`/`while`/`for`/`try` body) is the next unit if inheritance2 must go green.

---

## 16. Retyping inside function bodies — FIXED; inheritance2 GREEN (2026-06-12)

The dynamic-retype-in-function limitation from §15 is fixed (commit `ad5724ab12`).
`inheritance2` now verifies SUCCESSFUL.

Dynamic retyping (rebind a scalar-typed local to a string value or vice versa,
via a fresh symbol + `retype_aliases_` load redirect) was gated to
`block_nesting_ == 1` (module top level). It is actually sound on the whole
**unconditional spine** — the module body plus the chain of enclosing function
bodies — because there is no control-flow join there to leave the runtime type
ambiguous. Implementation: a new `function_body_depth_` counter, bumped only
when `get_block` converts a function body (`is_function_body=true`), and the
gate becomes `block_nesting_ == function_body_depth_ + 1`. An `if/while/for/try`
body adds a `block_nesting_` frame *without* a `function_body_depth_` frame, so
the equality fails and the conditional case stays refused (unsound at a join).
Fail-safe: an unrecognised block kind is treated as conditional (stricter, not
looser). `retype_aliases_` is keyed by the function-qualified symbol id, so
aliases never leak between functions — no cross-function clearing needed.

Validated: all retype-mechanism regressions hold (github_4770_retype_in_try —
retype in a `try` correctly stays refused; github_4774_*; float_cond_retype{,
-fail}; type-annotation-reassign-check*). New: retype_str_in_function{,_fail}
and retype_str_cond_gated (a str write inside a conditional must NOT retype).
No new failures across the control-flow / function / str / class family sweep.

**Still open (orthogonal, pre-existing — NOT this session's regressions):**
- `str(obj) == "..."` where `obj` is a migrated `Class*` instance and the class
  defines `__str__`: fails with `ERROR: Cannot compare non-function side
  effects`. `dispatch_unary_dunder_operator` (and the `str()` path) require a
  by-value struct operand, so `__str__` is not dispatched on a pointer instance.
  Confirmed pre-existing at commit `86c7c3003d` (before this session). Affects
  strings2, strings2_str_plain, strings2_str_multifield.
- Augmented attribute assignment `c.value += by` through a `Class*` parameter
  crashes in Converting (plain `c.value = v` is fine).
- Thread-subclass / module-scope construction (`threading_thread_subclass_*`).
