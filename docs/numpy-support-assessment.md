# ESBMC NumPy — Remaining Work

**Updated:** 2026-06-28.

This file tracks what is **not yet implemented or broken** in the NumPy
module. Features that already work are not listed here — check the
`regression/numpy/` suite and the git history for coverage.

---

## Recently completed

| Feature | PR / commit | Notes |
|---|---|---|
| Boolean array construction | This branch | `np.array([True, False, True])` works; bool type preserved |
| SMT store-chain for nested lists | This branch | `decompose_store_chain` / `decompose_select_chain` handle single-level chains gracefully |
| `reshape` / `ravel` / `flatten` | This branch | Constant-folded at JSON level; supports n-D; `-1` inferred dimension |
| dtype promotion table | This branch | `promote_numpy_dtype()` follows NumPy casting hierarchy |
| Variable-referenced shapes | This branch | `n = 3; np.zeros(n)` resolves at parse time |
| `linalg.inv` | This branch | 2×2 and 3×3; singular matrix detection |
| `linalg.solve` | This branch | 2×2 and 3×3 via matrix inverse; singular detection |
| `linalg.norm` | This branch | Frobenius norm for 1-D and 2-D arrays |
| n-D array support | This branch | Hard limit raised from 2-D to 8-D; `np.array()`, `np.zeros()`, `np.ones()`, `np.reshape()` all support n-D |

---

## Missing indexing / slicing

| Feature | Status | Notes |
|---|---|---|
| 2-D slicing `a[:,j]`, `a[i,:]` | Partially supported | Store-chain fix unblocks basic cases; full fancy indexing still missing |
| Boolean-mask indexing `a[mask]` | Missing | Bool arrays work but mask-based selection has no handler |
| Fancy/integer-array indexing `a[[0,2]]` | Missing | No handler |
| Strided slicing `a[::2]` | Untested | List slice model supports step but not tested for numpy |
| n-D tuple indexing `a[i,j,k]` | Missing | Need lowering for multi-axis tuple subscript |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | `empty` |
| Shape manipulation | `squeeze`, `stack`, `concatenate` |
| Sorting / searching | `sort`, `argsort`, `searchsorted`, `unique` |
| Statistics | `std`, `var`, `median`, `percentile` |
| Linear algebra | `linalg.eig`, `linalg.svd`; `det`/`inv`/`solve` limited to ≤3×3; `norm` limited to Frobenius |
| Type casting | `astype`; runtime dtype promotion for symbolic arrays |
| Random | `np.random.*` (all) |
| Structured arrays | Record dtypes |
| Views / strides | No aliasing model — all ops copy |
| Iteration | `nditer`, `flat` |

---

## Soundness concerns

1. **Constant-folding bypasses ESBMC's overflow/rounding checks** for the
   folded path. Use `--python-no-fold` to force SMT encoding and compare
   verdicts.
2. **`umath.c` float element-wise ops** are now typed, but symbolic array
   broadcasting still only works for concrete shapes.
3. **Scalability wall** (#5121): every array is a fully-unrolled value list.
   Large arrays explode. An SMT-array backing would fix this.

---

## KNOWNBUG tests

`dot6` (bool), `dot7`, `transpose2`, `transpose7` — silent wrong results.
Either model correctly or downgrade to explicit "unsupported".

---

## Prioritised next steps

1. **n-D tuple indexing** — lower `a[i,j,k]` to chained indexing for n-D arrays.
2. **Boolean-mask indexing** — `a[mask]` requires a runtime loop model.
3. **`astype` / runtime dtype casting** — needed for symbolic array dtype interop.
4. **`squeeze` / `stack` / `concatenate`** — shape manipulation completeness.
5. **`linalg.eig` / `linalg.svd`** — extend linear algebra beyond inv/solve/norm.
6. **SMT-array backing** — replace unrolled lists with SMT arrays for scalability.
7. **Symbolic shapes at runtime** — bounded-loop list construction for truly nondet sizes.
