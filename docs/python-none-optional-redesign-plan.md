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
