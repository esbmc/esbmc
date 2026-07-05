# ESBMC NumPy — Remaining Work

**Updated:** 2026-07-05.

This file tracks what is **not yet implemented or broken** in the NumPy
module. Completed items are in the git history and `regression/numpy/`.

---

## Recently completed

- **Array type preservation for `Any`-annotated subscript assignment**
  (`row = a[i]`, `col = a[:,j]`, `sel = a[[0,2]]`, `sel = a[mask]`) — the
  Python-side static annotator has no visibility into a numpy array's
  runtime shape, so a bare-variable assignment of a subscript result used
  to fall back to the uninformative `Any` type, storing the value
  generically (`void*`) instead of preserving the array type; a later
  `row[0]` either read a `NONDET` value or, for a 2-D result (e.g.
  `a[mask]` row selection), crashed the irep2 index constructor.  Fixed by
  `python_converter::resolve_any_subscript_array_type`
  (`converter_stmt.cpp`), which probes the RHS's real type when the
  annotation resolves to `Any` and the RHS is a `Subscript`, adopting the
  array type when the *source* array is at most 2-D (rejecting a 3-D+
  source explicitly — the resulting slice's nesting depth alone can't
  distinguish a legitimate 2-D row/column/fancy/mask selection from a
  3-D-derived one, so the check looks at the source instead). A
  fancy/mask/column-select result is already a materialized temp symbol
  and copies via a normal whole-array assignment; a plain `a[i]` chain is
  a raw index expression instead, which the backend cannot assign
  whole-array-to-whole-array (Z3 sort mismatch, same class of issue as the
  "whole-array assignment is not backend-safe" note below) — the final
  store copies it element by element instead
  (`any_subscript_array_needs_copy_`). A companion annotator fix
  (`python_annotation::resolve_subscript_type`,
  `annotation_conversion.inl`) was needed alongside: a literal index list
  (`a[[0, 2]]`) or a `Name` slice referencing a *list*-typed variable
  (`idx = [0, 2]; a[idx]`) was misclassified as a single-element scalar
  index (same code path as plain `a[i]`), annotating the result as a bare
  scalar instead of `Any`/list-typed and producing a confusing `'int'
  object is not subscriptable` error one level up.
- **Fancy indexing through a variable index list** (`idx = [0, 2];
  a[idx]`) — `idx` must resolve to a single, unambiguous literal
  assignment of concrete integers (mirroring the mask-reassignment guard
  below); resolved via `json_utils::find_var_decl` /
  `has_multiple_assignments_in_scope` and dispatched to the existing
  `python_list::build_fancy_index`, so all of its literal validation
  (integer-only, bounds-checked) and 2-D row-selection support apply
  unchanged (`converter_expr.cpp`).
- **`np.linalg.det`** — already fully implemented and tested (2×2, 3×3,
  negative values, non-square/ragged/complex/non-numeric rejection, near-
  singular, negative-zero, row-swap-sign) prior to this PR; the
  `docs/numpy-support-assessment.md` snapshot describing it as missing was
  stale (last touched in an earlier PR that didn't update this file). The
  only real gap found was the operational-model stub's signature
  (`models/numpy/linalg.py`), which declared `det(a: float, b: float)` —
  wrong shape for an API that takes one matrix-like argument; the stub's
  body is never executed (`det` dispatches entirely through
  `numpy_call_expr.cpp` before the model's Python body would run), so this
  was a documentation/API-surface fix only, not a behavioural one.
- **2-D slicing `a[:,j]` and `a[i,:]`** — column selection copies `row[j]`
  across every row of a fixed-shape 2-D array into a fresh 1-D array
  (`python_list::build_column_select`); row selection reuses the existing
  per-axis index chain (`a[i]` then a full `:` copy). Negative column
  indices are normalized the same way as 1-D indexing. Rejects any other
  slice/index combination (partial-bound slices, both axes sliced, 3+ dims)
  with the existing `multi-dimensional indexing ... not supported` error.
- **Fancy/integer-array indexing `a[[0,2]]`** — a literal index list is
  resolved and bounds-checked entirely at conversion time
  (`python_list::build_fancy_index`); each requested index must be a
  concrete integer literal (or its negation). Supports repeated indices.
  Extends to whole-row selection on a 2-D array (`a[[0,2]]` selecting rows)
  via column-by-column copies. Indexing through a *variable* holding an
  int array (rather than a literal list) is still not modelled — see
  below.
- **Boolean-mask indexing for 2-D arrays (whole-row selection)** —
  `a[mask]` on a 2-D array now selects whole rows, routed through
  `python_list::build_bool_mask_row_select`. This requires `mask` to
  resolve to a concrete boolean literal (`mask = np.array([True, False,
  ...])`, read back via its AST declaration) so the selected-row count is
  known at conversion time; the result is built as a fixed-size array,
  copied column by column. A symbolic (non-literal) mask is rejected
  explicitly rather than risking an unsound partial result — see below.
  1-D boolean-mask indexing (`build_bool_mask_index`) is unchanged. The
  mask lookup also rejects a mask variable that is reassigned anywhere in
  its scope (`json_utils::has_multiple_assignments_in_scope`), since the
  AST-declaration lookup used to read the mask's literal value returns the
  *first* textual assignment, not the one that actually reaches the
  subscript's use site — see "Code-review fixes" below.

### Code-review fixes (post-implementation)

A self-review pass (code-reviewer agent) on this PR's diff found three
issues, all fixed before landing:

1. **Stale mask read** — `build_bool_mask_row_select` resolved the mask's
   literal value via `find_var_decl`, which returns the first textual
   assignment to that name in scope, not the one reaching the subscript.
   A mask reassigned before use (`mask = ...; mask = ...; a[mask]`) would
   silently use the stale first value instead of erroring — a genuine
   soundness gap. Fixed by rejecting whenever the mask name has more than
   one assignment anywhere in scope
   (`json_utils::has_multiple_assignments_in_scope`, an existing helper
   already used the same way in `string_handler.cpp`), rather than
   attempting reaching-definitions analysis.
2. **`len()` misrouting scalar tuple indices** — the `len()` dispatch
   fix added for `len(a[i, :])` / `len(a[:, j])` matched *any*
   `Subscript` with a `Tuple`- or `List`-typed slice, including a fully
   scalar multi-index like `a[0, 1]` (which is a scalar, not an array).
   Fixed by checking the same `is_full_slice_node` condition used in
   `converter_expr.cpp`'s dispatch: only route to `kGetObjectSize` when
   exactly one of a 2-element tuple is a full slice (the two supported
   2-D slicing patterns); other tuple shapes fall through to the
   pre-existing (unrelated) scalar/strlen handling.
3. **Boolean literals in fancy indexing** — `a[[True, False]]` was
   silently accepted by `try_get_literal_int` as positional indices `[1,
   0]`, rather than being treated as (or rejected as) a boolean mask,
   producing wrong results for a plausible NumPy idiom. Fixed by
   excluding booleans from `try_get_literal_int`, so a boolean literal in
   a fancy-index list now raises the existing "fancy indexing only
   supports concrete integer indices" error instead of misinterpreting
   it.

### Implementation note: whole-array assignment is not backend-safe

Both whole-row selection paths (fancy indexing and boolean-mask) initially
tried to copy a selected row with a single `dst[k] = src[i]` assignment
(both sides array-typed). This is invalid in C (ESBMC's own C frontend
rejects `b[0]=a[0]` as "array type ... is not assignable") and produces a
Z3 sort mismatch (`Sorts (_ BitVec N) and (Array ...) are incompatible`)
when emitted directly as GOTO IR. Both paths now copy column by column
instead. Any future n-D indexing work that selects whole sub-arrays should
follow the same column-by-column pattern rather than a single array-level
assignment.

### Implementation note: whole-array assignment applies to plain subscript chains too

The Any-fix above hit the same backend limitation described in the note
above it: `row = a[i]` produces a raw index expression (not a
materialized temp symbol like the fancy/mask/column helpers already use),
so the final store cannot use a single whole-array-to-whole-array
`code_assignt` and instead copies element by element
(`any_subscript_array_needs_copy_` in `converter_stmt.cpp`).

---

## Missing indexing / slicing

| Feature | Status | Notes |
|---|---|---|
| Symbolic/non-literal boolean-mask row selection | Missing | `build_bool_mask_row_select` requires the mask to be a concrete literal (`np.array([True, False])`) resolved from its AST declaration; a mask built from nondet/computed values is rejected explicitly (no unsound fallback). |
| Strided slicing `a[::2]` | Untested | List slice model supports step but not tested for NumPy arrays |
| `a[i, j, k]` and n-D tuple indexing | Missing | Only up to 2-D indexing (single axis, or one axis sliced) is modelled; 3-D+ tuple indices are rejected explicitly. |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | `empty` |
| Sorting / searching | `sort`, `argsort`, `searchsorted`, `unique` |
| Statistics | `std`, `var`, `median`, `percentile` |
| Linear algebra | `inv`/`solve` limited to ≤3×3; `norm` limited to Frobenius; `eig`/`svd` limited to ≤3×3 concrete matrices |
| Random | `np.random.*` (all) |
| Structured arrays | Record dtypes |
| Views / strides | No aliasing model — all ops copy |
| Iteration | `nditer`, `flat` |

---

## Soundness concerns

1. **Constant-folding bypasses ESBMC's overflow/rounding checks** for the
   folded path. Use `--python-no-fold` to force SMT encoding and compare
   verdicts.
2. **Element-wise broadcasting** (e.g. `np.add(a, b)`) still requires
   concrete shapes at conversion time; symbolic shapes work only for array
   creation (`zeros`, `ones`, `full`).
3. **Scalability wall** (#5121): every array is a fully-unrolled value list.
   Large arrays explode. Symbolic shapes mitigate this via `--unwind` but do
   not eliminate the underlying state-explosion for large bounds.

---

## KNOWNBUG tests

`dot6` (bool), `dot7`, `transpose2`, `transpose7` — silent wrong results.
Either model correctly or downgrade to explicit "unsupported".

---

## Prioritised next steps

1. **`np.std` and `np.var`** — bounded loops using `build_list_at_call` +
   arithmetic expression builder; `np.mean` already exists. 4 tests each.
2. **Symbolic/non-literal boolean-mask row selection** — needs a design
   decision (see "Out of scope" below) before implementation.
3. **`a[i, j, k]` and n-D tuple indexing** — a larger frontend change; only
   worth picking up once the 2-D indexing surface above is exercised more
   in the field.

### Out of scope
- `np.random.*` — nondeterminism model requires a separate design decision.
- True SMT-array scalability — solver-level change; tracked in #5121.
- Views / strides / aliasing — deep model change; separate PR.
- Symbolic (non-literal) boolean-mask row selection — would need either a
  runtime-list model extension that can hold array-typed elements (blocked
  on the encoding gap described above) or a different result
  representation; needs a design decision before implementation.
