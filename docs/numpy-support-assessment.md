# ESBMC NumPy — Remaining Work

**Updated:** 2026-07-22.

This file tracks what is **not yet implemented or broken** in the NumPy
module. Completed items are in the git history and `regression/numpy/`.
Architectural decisions that gate specific pendencies here (referenced as
`ADR-NP-XXX`) are the normative source in `numpy-architecture-decisions.md`.

---

## Recently completed

- **Canonical bounded ndarray descriptor (initial slice)** — new
  `ndarray_descriptor` class (shape/strides/capacity/offset/dtype/buffer_id
  + invariant validation), the frontend-side scaffolding ADR-NP-001 and
  ADR-NP-003 are gated on. Wired to two consumers so far: a new `.ndim`
  attribute (previously entirely unsupported), and rejection of a negative
  array shape at creation (`np.zeros(-2)` now raises NumPy's actual
  `ValueError: negative dimensions are not allowed` instead of silently
  building an empty array or raising a misleading `TypeError`). The legacy
  nested-`array_typet` layout is still the sole runtime representation for
  everything else — this commit does not migrate consumers wholesale. See
  `regression/numpy/ndarray_descriptor_*`.
- **Symbolic (non-literal) boolean-mask *row* selection on 2-D arrays**
  (`a[mask]`, ADR-NP-001) — implemented via the canonical descriptor
  pattern: the result is a `{ rows: row_type[num_rows]; count: size_t }`
  struct built by a single runtime while-loop that scans the mask once,
  copies each selected row in order, and tracks a symbolic logical count
  (not the physical worst-case capacity). Indexing (`b[i]`, negative
  indices) is bounds-checked against the logical `count`, not the buffer's
  capacity. `.shape`/`.ndim` on the result read `(count, cols)` /
  `2`. Reassigned masks and masks with no local declaration (e.g. received
  as a parameter — see below) are also supported, since the symbolic path
  reads the mask's live runtime value rather than resolving it from its
  AST declaration. Six pre-existing regression tests that pinned the old
  "symbolic mask rejected" behaviour were updated from `_fail` to
  `_success`. See `regression/numpy/bool_mask_rows_*` and
  `regression/numpy/numpy_bool_mask_rows_*`.
- **Boolean-mask indexing through a function parameter**
  (`def f(a, mask): return a[mask]`) — a parameter playing either role in
  an `a[mask]` pattern (the array or the mask) now decays to
  pointer-to-*whole-array* instead of C's usual pointer-to-element/row, so
  the SUBSCRIPT converter can recognize it as a mask array at the call
  site (ordinary decay otherwise erases enough shape information that a
  1-D mask array is indistinguishable from a pointer to one bool). Also
  fixed a related soundness gap found while validating this: conflicting
  array shapes passed to the same parameter across different call sites
  used to be silently accepted (only the first-seen shape was checked);
  they are now rejected with `TypeError: conflicting array shapes...`.
  Forwarding through one intermediate function's own array/mask parameters
  is supported. See `regression/numpy/bool_mask_param_*` and
  `regression/numpy/array_param_mask_success`.
- **Identity array return from a function** (`def f(a): return a`) — a
  user function whose entire body is `return <param>` (or
  `return <param>[literal-index]` when that indexes to another array, e.g.
  a row of a 2-D array) is inlined at the call site to the caller's own
  argument expression, since arrays still aren't a valid by-value return
  type in the current GOTO model (see "Missing indexing/slicing" below).
  This is a narrow, call-site-local fix, not a general return mechanism —
  see the note on the general case below. See
  `regression/numpy/array_return_identity_success` and
  `array_return_empty_edge`.
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
  *row-select* path (`a[mask]` selecting whole rows) is now also supported
  symbolically — see above.
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
  parameter otherwise inferred as array-shaped). Boolean-mask indexing
  through a parameter is now also supported — see above. Returning a numpy
  array *out* of a function by value (beyond the narrow identity case — see
  above) is a separate, still-unsupported case — see "Missing indexing /
  slicing" below. See `regression/numpy/numpy_param_array_*` and
  `array_param_*`.
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
| Returning a numpy array *out* of a function by value (general case: a sub-array, e.g. `def f(a): return a[0]`, or any non-trivial body) | Missing | Arrays aren't valid by-value return types in the current GOTO model. Only the narrow *identity*-return case is fixed (see "Recently completed") — the general case was attempted twice this round and reverted both times after hitting the same structural wall: **(1)** inlining the substituted return expression at every call site works for the eligibility check but the two-pass assignment machinery (`create_symbol_for_unannotated_assign` type-probes the RHS once, then `get_var_assign` converts it again "for real") evaluates a `Subscript` return expression twice, duplicating the bounds-check GOTO code it emits and corrupting the result (confirmed via `--goto-functions-only`: the second, discarded evaluation's DECLs still land in the block); a cross-call-node cache (keyed by the AST node's address, confirmed to see the same `current_block` on both hits) prevented the double-conversion but did *not* fix the wrong result, meaning the bug is elsewhere in that pipeline. **(2)** A single-member wrapper-struct return type (`struct { value: array_type }`, unwrapped right after a real, once-only function call via the existing `store_call_result` helper — mirroring how a returned tuple already works today) also hit a variant of the same issue: while building it, a separate pre-existing bug was found and fixed (a static Python-level pre-pass injects a wrong `-> Any` return annotation for `return a[0]`, decided *before* parameters are processed, which pre-empted the array-shape detection with a `double` default — now deferred to the post-body GOTO scan, which sees real converted types), but the wrapped struct still isn't reliably unwrapped before a variable's type is decided elsewhere in the same assignment pipeline, causing a segfault (`build_index` dereferencing a struct's `.subtype()`). Both attempts point at the same root cause: `y = f(...)`'s type is decided by more than one code path in `get_var_assign`/`create_symbol_for_unannotated_assign`, not uniformly from what `get_expr` returns. A real fix needs to understand and consolidate that pipeline first, not just work around it at the call site. |
| Strided column slice combined with explicit bounds (`a[:, 1:3:2]`) | Missing | `build_strided_column_select` only models the bare-step form (`a[:, ::step]`); combining a column step with explicit bounds is rejected explicitly, since the result width would then need to be resolved from runtime bounds instead of the array's static shape. |
| Bounded/partial slice or more than one slice axis mixed with integer indices in one tuple (`a[0:2, 0, 0]`, `a[:, :, 0]`) | Missing | Rejected explicitly (`TypeError: multi-dimensional indexing ... numpy arrays are modelled as 1D lists`); the single-full-slice-axis case (`a[:, 0, 0]`) is supported — see "Recently completed". |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | `empty`; `empty_like`/`zeros_like`/`ones_like`/`full_like` |
| Sorting / searching | `sort`, `argsort`, `searchsorted`, `unique` |
| Statistics | `median`, `percentile` |
| Linear algebra | `inv`/`solve` limited to ≤3×3; `norm` limited to Frobenius; `eig`/`svd` limited to ≤3×3 concrete matrices |
| Random | `np.random.*` (all) |
| Structured arrays | Record dtypes |
| Views / strides | No aliasing model — all ops copy; see "Soundness concerns" #4, this is a confirmed unsound gap, not just a missing feature |
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
4. **Basic-indexing views are silently copied, not aliased — confirmed with
   a concrete false-positive counterexample** (ADR-NP-003). NumPy's basic
   indexing (`a[0]`) returns a *view* sharing the source's buffer; this
   frontend always copies. Reproduced with no function call involved at
   all:
   ```python
   import numpy as np
   x = np.array([[1, 2], [3, 4]])
   row = x[0]
   row[0] = 999
   assert x[0][0] == 1  # numpy: fails (row aliases x). ESBMC today: VERIFICATION SUCCESSFUL.
   ```
   ESBMC reports `VERIFICATION SUCCESSFUL` — a silently wrong verdict per
   the ADR's own principle ("a subapproximation that could hide a bug is
   not acceptable"). This is not a missing-feature gap, it's an existing,
   reproducible unsoundness, and it predates and is independent of this
   round's work (found while investigating whether a returned view needed
   its own rejection — it turns out the *non-returned* case is equally
   unsound today). See ADR-NP-003 "Etapa 1: protecao de solidez" for the
   intended fix shape (reject conservatively on write-to-view,
   write-to-source-with-live-view, or escape, until the definitive
   descriptor-based model lands).

---

## KNOWNBUG tests

`dot6` (bool), `dot7`, `transpose2`, `transpose7` — silent wrong results.
Either model correctly or downgrade to explicit "unsupported".

---

## Prioritised next steps

1. **View/aliasing soundness protection** (ADR-NP-003 "Etapa 1") — the
   highest-priority item: "Soundness concerns" #4 above is a confirmed,
   reproducible false-`VERIFICATION SUCCESSFUL` with no function call or
   other feature involved (`row = x[0]; row[0] = 999; assert x[0][0] == 1`
   should fail and doesn't). Needs conservative rejection of write-to-view,
   write-to-source-with-a-live-view, and view escape, until the definitive
   descriptor-based model (buffer_id/offset/strides shared between a view
   and its source) lands. The canonical descriptor scaffolding from
   "Recently completed" is a starting point but isn't wired into the
   general indexing/assignment path yet.
2. **NumPy arrays as function return values, general case** — only the
   narrow identity-return pattern (`return <param>` or
   `return <param>[literal-index]`) is fixed; a sub-array return from a
   non-trivial function body (e.g. `def f(a): return a[0]` when `0` isn't
   simply substitutable, or any function with more than the one return
   statement) is still unsupported. Two implementation strategies were
   tried and reverted this round — see the "Missing indexing / slicing"
   table entry above for the detailed failure analysis. Both hit the same
   root cause: `y = f(...)`'s type is decided by more than one code path
   in the unannotated-assignment machinery
   (`get_var_assign`/`create_symbol_for_unannotated_assign`), not
   uniformly from what `get_expr` returns for the call. A real fix likely
   needs to understand and consolidate that pipeline first, rather than
   work around it purely at the call site or the callee's return
   statement.
3. **Strided column slice combined with explicit bounds** (`a[:, 1:3:2]`) —
   currently rejected explicitly; the bare-step form (`a[:, ::step]`) is
   supported. Would need the result width resolved from runtime bounds
   instead of the array's static shape.
4. **Bounded/partial slice or multiple slice axes mixed with integer
   indices in one tuple** (`a[0:2, 0, 0]`, `a[:, :, 0]`) — currently
   rejected explicitly; the single-full-slice-axis case is supported.
5. **`np.empty`, sorting/searching (`sort`/`argsort`/`searchsorted`/
   `unique`), statistics (`median`/`percentile`), `np.random.*`, `flat`/
   `nditer`, and the `dot6`/`dot7`/`transpose2`/`transpose7` KNOWNBUGs** —
   entirely untouched this round; see "Missing API surface" and "KNOWNBUG
   tests" above. `np.random.*` needs ADR-NP-002 (accepted, implementation
   pending).

### Out of scope
- True SMT-array scalability — `array_typet` already lowers to SMT
  select/store; see ADR-NP-004. Any further change is benchmark-gated.
- Extending the runtime-list model to hold array-typed elements — rejected
  as an approach for symbolic 2-D boolean-mask row selection (see
  ADR-NP-001's "Alternativas rejeitadas"); still considered
  disproportionately risky for any other use case.
