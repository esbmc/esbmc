# ESBMC NumPy — Remaining Work

**Updated:** 2026-06-27.

This file tracks what is **not yet implemented or broken** in the NumPy
module. Features that already work are not listed here — check the
`regression/numpy/` suite and the git history for coverage.

---

## Blocking bugs (must fix before further indexing work)

| Bug | Symptom | Root cause |
|---|---|---|
| Boolean array construction crash | `np.array([True, False, True])` segfaults | `migrate_type_back` receives an unrepresentable bool-list type from `build_push_list_call` → `get_list_element_info` |
| 2-D nested-list element access | `m[0:2][0][0]` crashes in solver | `smt_solver.cpp:decompose_store_chain` assertion failure on nested store chains |

---

## Missing indexing / slicing

| Feature | Status | Notes |
|---|---|---|
| 2-D slicing `a[:,j]`, `a[i,:]` | Unsupported | Frontend lowering exists but blocked by SMT store-chain bug above |
| Boolean-mask indexing `a[mask]` | Missing | Blocked by bool-array construction crash above |
| Fancy/integer-array indexing `a[[0,2]]` | Missing | No handler |
| Strided slicing `a[::2]` | Untested | List slice model supports step but not tested for numpy |
| 3-D+ indexing | Unsupported by design | Hard 2-D ceiling |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | `empty` |
| Shape manipulation | `reshape`, `ravel`, `flatten`, `squeeze`, `stack`, `concatenate` |
| Sorting / searching | `sort`, `argsort`, `searchsorted`, `unique` |
| Statistics | `std`, `var`, `median`, `percentile` |
| Linear algebra | `linalg.inv`, `linalg.solve`, `linalg.eig`, `linalg.svd`, `linalg.norm`; `det` limited to ≤3×3 |
| Type casting | `astype`; full NumPy promotion table |
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

1. **Fix bool-array construction crash** — unblocks boolean-mask indexing.
2. **Fix SMT store-chain for nested lists** — unblocks 2-D slicing.
3. **`reshape` / `ravel` / `flatten`** — unlocks tensor-style code.
4. **Symbolic shapes / dtype promotion** — allows shape values that are
   symbolic within a static bound.
5. **`linalg.{inv,solve,norm}`** — extend beyond `det`.
6. **Lift 2-D ceiling** — requires flat backing + reshape + n-D indexing.
