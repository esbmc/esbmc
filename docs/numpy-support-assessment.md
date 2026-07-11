# ESBMC NumPy — Remaining Work

**Updated:** 2026-07-11.

This file tracks what is **not yet implemented or broken** in the NumPy
module. Completed items are in the git history and `regression/numpy/`.

---

## Recently completed

- **`np.std` and `np.var`** — 1-D and 2-D (flattened) concrete numeric
  inputs; rejects empty/non-numeric input and `axis`/`ddof`/`keepdims`/
  `where`/`out`/`dtype` kwargs explicitly. `std` is `sqrt(var)` on the same
  code path. See `regression/numpy/numpy_std_*` and `numpy_var_*`.
- **Symbolic (non-literal) boolean-mask selection on 1-D arrays** —
  confirmed already sound and now covered by regression tests: a mask built
  from nondet/computed values works via the existing runtime while-loop
  path (`build_bool_mask_index`), including through reassignment, since it
  reads the mask's current value at the point of use rather than folding it
  statically. This also covers `a[i][mask]` (a row sliced off a 2-D array,
  then filtered). See `regression/numpy/bool_mask_symbolic_*`. The 2-D
  *row-select* path (`a[mask]` selecting whole rows) still requires a
  literal mask — see "Missing indexing / slicing" below.
- **`a[i, j, k]` and n-D tuple indexing** — confirmed already implemented
  for literal/negative/symbolic integer indices on 3-D+ arrays, including
  out-of-bounds bounds-checking; the assessment above was stale. Only
  mixing a slice with integer indices in the same tuple (`a[:, 0, 0]`)
  remains explicitly unsupported. See `regression/numpy/tuple_index_3d_*`.
- **NumPy arrays as function parameters (soundness fix)** — passing a numpy
  array into a user-defined function used to silently reinterpret the
  array's raw bytes as the heap `PyListObject` struct backing Python's
  `list` (the default parameter representation), producing wrong
  verification results or alignment faults instead of a real bug. This is
  now rejected explicitly with a clear `TypeError` at the call boundary
  instead. Genuine support (making the parameter usable inside the callee)
  is still open — see "Prioritised next steps".

---

## Missing indexing / slicing

| Feature | Status | Notes |
|---|---|---|
| Symbolic/non-literal boolean-mask *row* selection (`a[mask]` on a 2-D array) | Missing | `build_bool_mask_row_select` requires the mask to be a concrete literal (`np.array([True, False])`) resolved from its AST declaration; a mask built from nondet/computed values is rejected explicitly (no unsound fallback). The 1-D case is supported — see "Recently completed". |
| Strided slicing `a[::2]` | Untested | List slice model supports step but not tested for NumPy arrays |
| Mixing a slice with integer indices in one tuple index (`a[:, 0, 0]`) | Missing | Rejected explicitly (`TypeError: multi-dimensional indexing ... numpy arrays are modelled as 1D lists`); pure integer/negative n-D tuple indexing itself is supported — see "Recently completed". |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | `empty` |
| Sorting / searching | `sort`, `argsort`, `searchsorted`, `unique` |
| Statistics | `median`, `percentile` |
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

1. **NumPy arrays as genuine function parameters** — the silent-unsoundness
   bug is closed (see "Recently completed"), but the array is still
   unusable inside the callee (calls are rejected outright). Needs either a
   distinct parameter representation for numpy arrays (decay to a raw
   element pointer + size, not `PyListObject*`) or a documented annotation
   convention (e.g. an `np.ndarray` type hint, currently unparsed by the
   frontend) so the callee's parameter type can be inferred soundly. This
   is the next PR to pick up.
2. **Symbolic/non-literal boolean-mask *row* selection** (2-D `a[mask]`) —
   still needs the design decision described in "Out of scope" below
   (blocked on the runtime-list-of-arrays encoding gap).
3. **Mixing slices with integer indices in one tuple index** (`a[:, 0, 0]`)
   — currently rejected explicitly; would need the slice axis to produce a
   sub-array result alongside the integer-indexed axes.

### Out of scope
- `np.random.*` — nondeterminism model requires a separate design decision.
- True SMT-array scalability — solver-level change; tracked in #5121.
- Views / strides / aliasing — deep model change; separate PR.
- Symbolic (non-literal) boolean-mask *row* selection — would need either a
  runtime-list model extension that can hold array-typed elements (blocked
  on the encoding gap described above) or a different result
  representation; needs a design decision before implementation.
