# ESBMC NumPy — Remaining Work

**Updated:** 2026-07-02.

This file tracks what is **not yet implemented or broken** in the NumPy
module. Completed items are in the git history and `regression/numpy/`.

---

## Recently completed

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

### Implementation note: pre-existing gap surfaced by this work

Assigning *any* subscript result whose static type is a fixed-size array
(not just the new 2-D slicing/fancy/mask results — also the pre-existing
`row = a[i]` chained-index case) to a **bare, unannotated variable**
produces `NONDET` assertions (or, for nested 2-D results, an internal
assertion failure) instead of a sound value. Root cause: the Python-side
static annotator has no visibility into a numpy array's runtime shape, so
it falls back to the uninformative `Any` type for such assignments, and
`get_var_assign`'s `Any`-annotated path stores the value generically
(effectively `void*`) rather than preserving the array type. This is
**pre-existing and independent of this PR's features** (reproduces with
plain `row = a[i]` on master before this change); all regression tests
added by this PR avoid it by chaining the subscript inline (e.g.
`a[i, :][0]` instead of `row = a[i, :]; row[0]`) rather than fixing the
underlying type-inference gap, which is a separate, larger change. See
"Prioritised next steps" below.

---

## Missing indexing / slicing

| Feature | Status | Notes |
|---|---|---|
| Assigning an array-typed subscript result to a bare variable | Missing | `row = a[i]` / `row = a[i, :]` / `row = a[[0,2]]` / `row = a[mask]` all lose their type via the `Any`-annotation fallback; works only when the subscript is used inline. See implementation note above. |
| Fancy indexing through a variable (`idx = [0, 2]; a[idx]`) | Missing | Only a literal index list at the subscript site (`a[[0, 2]]`) is resolved; a `Name` holding an int array still raises the explicit "fancy indexing with a non-boolean array" error. |
| Symbolic/non-literal boolean-mask row selection | Missing | `build_bool_mask_row_select` requires the mask to be a concrete literal (`np.array([True, False])`) resolved from its AST declaration; a mask built from nondet/computed values is rejected explicitly (no unsound fallback). |
| Strided slicing `a[::2]` | Untested | List slice model supports step but not tested for NumPy arrays |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | `empty` |
| Sorting / searching | `sort`, `argsort`, `searchsorted`, `unique` |
| Statistics | `std`, `var`, `median`, `percentile` |
| Linear algebra | `det`; `inv`/`solve` limited to ≤3×3; `norm` limited to Frobenius; `eig`/`svd` limited to ≤3×3 concrete matrices |
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
4. **Bare-variable assignment of array-typed subscript results** loses type
   information (see implementation note above) — a real but pre-existing
   soundness/usability gap, not introduced by this PR.

---

## KNOWNBUG tests

`dot6` (bool), `dot7`, `transpose2`, `transpose7` — silent wrong results.
Either model correctly or downgrade to explicit "unsupported".

---

## Prioritised next steps

1. **Fix the `Any`-annotation fallback for array-typed subscript results**
   (see implementation note above) — `get_var_assign`'s reconciliation for
   `current_element_type == any_type()` already special-cases a `char*`
   RHS; extending it to array/pointer-to-array RHS is the natural fix, but
   the *declared* type must land as a plain array (not a decayed
   pointer-to-array) for downstream subscripting to resolve correctly —
   the naive version of this fix was reverted during this PR after it
   produced a `pointer(array)`-typed symbol that broke subscript dispatch.
   This unblocks `row = a[i, :]` / `row = a[[0,2]]` / `row = a[mask]` style
   code, which today only works when the subscript is inlined.
2. **Fancy indexing through a variable index list** (`idx = [0, 2];
   a[idx]`) — extend `build_fancy_index`'s literal-list requirement to also
   accept a `Name` reference whose declaration is a literal int list,
   mirroring how `build_bool_mask_row_select` reads a mask's declaration.
3. **`linalg.det`** — 2×2 = `ad-bc`, 3×3 = cofactor expansion; same pattern
   as `linalg.inv`. 3 regression tests.
4. **`np.std` and `np.var`** — bounded loops using `build_list_at_call` +
   arithmetic expression builder; `np.mean` already exists. 4 tests each.

### Out of scope
- `np.random.*` — nondeterminism model requires a separate design decision.
- True SMT-array scalability — solver-level change; tracked in #5121.
- Views / strides / aliasing — deep model change; separate PR.
- Symbolic (non-literal) boolean-mask row selection — would need either a
  runtime-list model extension that can hold array-typed elements (blocked
  on the encoding gap described above) or a different result
  representation; needs a design decision before implementation.
