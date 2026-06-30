# ESBMC NumPy — Remaining Work

**Updated:** 2026-06-30.

This file tracks what is **not yet implemented or broken** in the NumPy
module. Features that already work are not listed here — check the
`regression/numpy/` suite and the git history for coverage.

---

## Recently completed

| Feature | PR / commit | Notes |
|---|---|---|
| Boolean array construction | Previous branch | `np.array([True, False, True])` works; bool type preserved |
| SMT store-chain for nested lists | Previous branch | `decompose_store_chain` / `decompose_select_chain` handle single-level chains gracefully |
| `reshape` / `ravel` / `flatten` | Previous branch | Constant-folded at JSON level; supports n-D; `-1` inferred dimension |
| dtype promotion table | Previous branch | `promote_numpy_dtype()` follows NumPy casting hierarchy |
| Variable-referenced shapes | Previous branch | `n = 3; np.zeros(n)` resolves at parse time |
| `linalg.inv` | Previous branch | 2×2 and 3×3; singular matrix detection |
| `linalg.solve` | Previous branch | 2×2 and 3×3 via matrix inverse; singular detection |
| `linalg.norm` | Previous branch | Frobenius norm for 1-D and 2-D arrays |
| n-D array support | Previous branch | Hard limit raised from 2-D to 8-D; `np.array()`, `np.zeros()`, `np.ones()`, `np.reshape()` all support n-D |
| **n-D tuple indexing** `a[i,j,k]` | `feat/numpy-indexing-shape-linalg` | Generic loop over all tuple axes; rank-mismatch → `IndexError` |
| **Boolean-mask indexing** `a[mask]` | `feat/numpy-indexing-shape-linalg` | 1-D static arrays; bounded while-loop; compile-time length check |
| **`astype` / runtime dtype casting** | `feat/numpy-indexing-shape-linalg` | `ndarray.astype("float64")` etc.; element-wise cast via `build_typecast` |
| **`squeeze` / `stack` / `concatenate`** | `feat/numpy-indexing-shape-linalg` | Shape manipulation completeness for 1-D static arrays |
| **`linalg.eig` / `linalg.svd`** | `feat/numpy-indexing-shape-linalg` | Small concrete matrices (≤3×3); eigenvalue/singular-value symbolic assertion |
| **Per-iteration loop overhead reduction** | `feat/numpy-indexing-shape-linalg` | Unsigned-index fast path in `build_list_at_call` skips dead negative-index machinery |
| **Symbolic shapes at runtime** | `feat/numpy-indexing-shape-linalg` | `np.zeros(n)`, `np.ones(n)`, `np.full(n, v)` for nondet/symbolic `n`; bounded while-loop fill; `--unwind` controls bound |

---

## Missing indexing / slicing

| Feature | Status | Notes |
|---|---|---|
| 2-D slicing `a[:,j]`, `a[i,:]` | Partial | Basic store-chain cases work; full column/row slice still missing |
| Fancy/integer-array indexing `a[[0,2]]` | Missing | No handler |
| Strided slicing `a[::2]` | Untested | List slice model supports step but not tested for NumPy arrays |
| Boolean-mask for 2-D arrays / list-backed masks | Missing | Current `build_bool_mask_index` only handles 1-D `array_typet` masks |

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
2. **`umath.c` float element-wise ops** are now typed, but symbolic array
   broadcasting still only works for concrete shapes.
3. **Scalability wall** (#5121): every array is a fully-unrolled value list.
   Large arrays explode. Symbolic shapes mitigate this via `--unwind` but do
   not eliminate the underlying state-explosion for large bounds.

---

## KNOWNBUG tests

`dot6` (bool), `dot7`, `transpose2`, `transpose7` — silent wrong results.
Either model correctly or downgrade to explicit "unsupported".

---

## Next PR plan

### Goal
Close the most impactful remaining gaps in indexing and linear algebra
coverage. Target: 2-D column/row slicing and `linalg` expansion.

### Proposed scope

1. **2-D slicing `a[:,j]` and `a[i,:]`** (high impact, unblocks many real programs)
   - Detect `Slice(lower=None, upper=None, step=None)` in a tuple subscript as a
     "select all" axis; emit a bounded loop that copies the selected column/row into
     a fresh 1-D array.
   - Requires at least 4 new regression tests (2 pass, 1 fail shape-mismatch, 1 edge
     boundary).

2. **Fancy / integer-array indexing `a[[0,2,4]]`** (medium impact)
   - Detect subscript whose type is `array_typet` with integer element type; build a
     bounded loop over the index array and `list_at` the source.
   - 4 regression tests minimum.

3. **`linalg.det`** (quick win — formula already available for ≤3×3)
   - 2×2 = `ad-bc`; 3×3 = cofactor expansion. Same pattern as `linalg.inv`.
   - 3 regression tests (2×2 pass, 3×3 pass, singular-matrix pass).

4. **Expand `linalg.eig` / `linalg.svd` to 4×4** (medium effort)
   - Extend the closed-form / iterative approach up to 4×4; add `--unwind` guidance
     to test.desc for tests that require more loop iterations.

5. **Statistics: `np.std` and `np.var`** (straightforward given `np.mean` exists)
   - Express as bounded loops over the array; use existing `build_list_at_call` +
     arithmetic expression builder.
   - 4 regression tests each.

6. **Boolean-mask indexing for 2-D arrays** (extends phase-2 from this PR)
   - Handle the case where source is a 2-D `array_typet`; mask selects rows.
   - 4 regression tests.

### Out of scope for next PR
- `np.random.*` — nondeterminism model requires separate design decision.
- True SMT-array scalability — requires solver-level changes (quantifiers or
  uninterpreted functions); tracked separately in #5121.
- Views / strides / aliasing — deep model change; separate PR.
