# ESBMC NumPy — Remaining Work

**Updated:** 2026-07-01.

This file tracks what is **not yet implemented or broken** in the NumPy
module. Completed items are in the git history and `regression/numpy/`.

---

## Missing indexing / slicing

| Feature | Status | Notes |
|---|---|---|
| 2-D slicing `a[:,j]`, `a[i,:]` | Partial | Basic store-chain cases work; full column/row slice still missing |
| Fancy/integer-array indexing `a[[0,2]]` | Missing | No handler |
| Strided slicing `a[::2]` | Untested | List slice model supports step but not tested for NumPy arrays |
| Boolean-mask for 2-D arrays / list-backed masks | Missing | `build_bool_mask_index` only handles 1-D `array_typet` masks |

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

---

## KNOWNBUG tests

`dot6` (bool), `dot7`, `transpose2`, `transpose7` — silent wrong results.
Either model correctly or downgrade to explicit "unsupported".

---

## Next PR — proposed scope

1. **2-D slicing `a[:,j]` and `a[i,:]`** — detect `Slice(lower=None,
   upper=None, step=None)` in a tuple subscript; emit a bounded loop that
   copies the selected column/row into a fresh 1-D array. 4 regression tests.

2. **Fancy / integer-array indexing `a[[0,2,4]]`** — detect subscript whose
   type is `array_typet` with integer element type; bounded loop over the
   index array with `list_at` on the source. 4 regression tests.

3. **`linalg.det`** — 2×2 = `ad-bc`, 3×3 = cofactor expansion; same pattern
   as `linalg.inv`. 3 regression tests.

4. **`np.std` and `np.var`** — bounded loops using `build_list_at_call` +
   arithmetic expression builder; `np.mean` already exists. 4 tests each.

5. **Boolean-mask indexing for 2-D arrays** — `mask` selects whole rows;
   extends `build_bool_mask_index` to handle `array_typet` sources. 4 tests.

### Out of scope
- `np.random.*` — nondeterminism model requires a separate design decision.
- True SMT-array scalability — solver-level change; tracked in #5121.
- Views / strides / aliasing — deep model change; separate PR.
