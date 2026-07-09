# ESBMC NumPy — Remaining Work

**Updated:** 2026-07-05.

This file tracks what is **not yet implemented or broken** in the NumPy
module. Completed items are in the git history and `regression/numpy/`.

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
4. **NumPy arrays as function parameters** are not well modelled: a numpy
   array crossing a function boundary decays to pointer-to-array, and even
   a plain 2-D `row = a[0]` inside the callee currently fails (observed as
   a dereference failure), not just the already-excluded 3-D+ case.

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
3. **NumPy arrays as function parameters** — investigate the
   pointer-to-array decay at function boundaries (see soundness concern
   above); needed before any n-D indexing work can safely extend past
   module-level arrays.
4. **`a[i, j, k]` and n-D tuple indexing** — a larger frontend change; only
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
