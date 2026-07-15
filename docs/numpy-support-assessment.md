# ESBMC NumPy — Remaining Work

**Updated:** 2026-07-15.

This file tracks what is **not yet implemented or broken** in the NumPy
module. Completed items are in the git history and `regression/numpy/`.
Architectural decisions that gate specific pendencies here (referenced as
`ADR-NP-XXX`) are the normative source in `numpy-architecture-decisions.md`.

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
  literal mask — see "Missing indexing / slicing" below (ADR-NP-001).
- **`a[i, j, k]` and n-D tuple indexing** — confirmed already implemented
  for literal/negative/symbolic integer indices on 3-D+ arrays, including
  out-of-bounds bounds-checking; the assessment above was stale. Mixing a
  slice with integer indices in the same tuple (`a[:, 0, 0]`) is now also
  supported — see the dedicated entry below. See
  `regression/numpy/tuple_index_3d_*`.
- **NumPy arrays as genuine function parameters** — a numpy array passed
  into a user-defined function now keeps a concrete array type (inferred
  from the shapes its callers actually pass, including through one level of
  forwarding via another function's own array parameter) instead of
  decaying to `PyListObject*`/`Any`, so it stays indexable inside the
  callee. Parameters whose call sites can't be resolved this way keep the
  old default and the existing explicit-rejection/boundary diagnostics
  still fire for genuine mismatches (e.g. a scalar argument against a
  parameter otherwise inferred as array-shaped, or boolean-mask indexing
  through a parameter, which is out of scope for this fix). Returning a
  numpy array *out* of a function by value is a separate, still-unsupported
  case — see "Missing indexing / slicing" below. See
  `regression/numpy/numpy_param_array_*` and `array_param_*`.
- **Mixing a slice with integer indices in one tuple index** (`a[:, 0, 0]`,
  `a[0, :, 0]`) — an N-D tuple subscript with exactly one full-slice axis
  (`:`) and every other axis a literal/resolvable integer now lowers to a
  bounded copy along the slice axis, generalizing the existing 2-D
  column-select path. A bounded/partial slice (`a[0:2, 0, 0]`) or more than
  one slice axis (`a[:, :, 0]`) in the same tuple stays rejected explicitly.
  See `regression/numpy/numpy_tuple_mixed_slice_*`.
- **Strided slicing (`a[::2]`, `a[1::2]`, `a[::-1]`)** — confirmed already
  supported and now covered by regression tests for 1-D arrays (the
  existing slice model already implemented `step`). Extended to 2-D:
  `a[::2, :]` (strided row selection) and `a[:, ::2]` (strided column
  selection, bare step only — see "Missing indexing / slicing"). `step=0`
  continues to raise `ValueError` at runtime. See
  `regression/numpy/numpy_strided_slice_*`.

---

## Missing indexing / slicing

| Feature | Status | Notes |
|---|---|---|
| Symbolic/non-literal boolean-mask *row* selection (`a[mask]` on a 2-D array) | Missing | `build_bool_mask_row_select` requires the mask to be a concrete literal (`np.array([True, False])`) resolved from its AST declaration; a mask built from nondet/computed values is rejected explicitly. A bounded-result-plus-explicit-count fallback (worst-case-sized result + a runtime `count`, unused slots left as padding) was prototyped and reverted: the count lives outside the result's type, so `len`/`shape` and any other consumer would observe the physical capacity instead of the logical size — `ADR-NP-001` (`numpy-architecture-decisions.md`) classifies this as unsound (violates the "shape is part of the modelled value" principle) and requires the canonical ndarray descriptor (buffer/shape/strides/offset) before this can be implemented soundly. The 1-D case is supported — see "Recently completed". |
| Returning a numpy array *out* of a function by value | Missing | Arrays aren't valid by-value return types in the current GOTO model (confirmed empirically: the backend rejects the array-typed return with "Can't construct rvalue reference to array type during dereference"). A function may still receive an array parameter and return a *scalar* derived from it (e.g. `a[i][j]`) — only returning the array/a sub-array itself is blocked. |
| Boolean-mask indexing through a function parameter (`def f(a, mask): return a[mask]`) | Missing | Rejected explicitly at the call boundary (parameter keeps its old `Any`/`PyListObject*` default) rather than attempting the array-parameter inference, since combining the two features wasn't validated together in this PR. |
| Strided column slice combined with explicit bounds (`a[:, 1:3:2]`) | Missing | `build_strided_column_select` only models the bare-step form (`a[:, ::step]`); combining a column step with explicit bounds is rejected explicitly, since the result width would then need to be resolved from runtime bounds instead of the array's static shape. |
| Bounded/partial slice or more than one slice axis mixed with integer indices in one tuple (`a[0:2, 0, 0]`, `a[:, :, 0]`) | Missing | Rejected explicitly (`TypeError: multi-dimensional indexing ... numpy arrays are modelled as 1D lists`); the single-full-slice-axis case (`a[:, 0, 0]`) is supported — see "Recently completed". |

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

1. **Canonical bounded ndarray descriptor** (buffer/capacity/shape/strides/
   offset/rank/dtype/buffer_id, see `numpy-architecture-decisions.md`) —
   blocks ADR-NP-001 (symbolic 2-D boolean-mask row selection) and the
   definitive model for ADR-NP-003 (views/aliasing). This is the
   foundational next PR: several other items on this list are gated on it.
2. **Symbolic/non-literal boolean-mask *row* selection** (2-D `a[mask]`) —
   ADR-NP-001; needs the canonical descriptor above so the result's logical
   row count is part of the modelled value instead of a detached runtime
   counter.
3. **NumPy arrays as function return values** — a function can receive an
   array parameter and use/forward it, but cannot return the array (or a
   sub-array) itself by value; only a scalar derived from it. Needs either
   a distinct by-reference return representation (the caller passes a
   destination pointer, mirroring the parameter decay already used) or a
   small wrapper-struct return type, since arrays aren't valid by-value
   return types in the current GOTO model.
4. **Boolean-mask indexing through a function parameter**
   (`def f(a, mask): return a[mask]`) — currently rejected explicitly at the
   call boundary rather than combined with the array-parameter inference
   from this PR; needs the two features validated together.
5. **Strided column slice combined with explicit bounds** (`a[:, 1:3:2]`) —
   currently rejected explicitly; the bare-step form (`a[:, ::step]`) is
   supported. Would need the result width resolved from runtime bounds
   instead of the array's static shape.
6. **Bounded/partial slice or multiple slice axes mixed with integer
   indices in one tuple** (`a[0:2, 0, 0]`, `a[:, :, 0]`) — currently
   rejected explicitly; the single-full-slice-axis case is supported.

### Out of scope
- `np.random.*` — nondeterminism model requires ADR-NP-002 (accepted,
  implementation pending).
- True SMT-array scalability — `array_typet` already lowers to SMT
  select/store; see ADR-NP-004. Any further change is benchmark-gated.
- Views / strides / aliasing — ADR-NP-003; soundness-protection stage
  first, definitive model gated on the canonical descriptor.
- Extending the runtime-list model to hold array-typed elements — rejected
  as an approach for symbolic 2-D boolean-mask row selection (see
  ADR-NP-001's "Alternativas rejeitadas"); still considered
  disproportionately risky for any other use case.
