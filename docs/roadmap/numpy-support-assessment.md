# ESBMC NumPy — Remaining Work

**Updated:** 2026-08-02.

This file tracks what is **not yet implemented or broken** in the NumPy
module. Completed items are in the git history and `regression/numpy/`.
Architectural decisions that gate specific pendencies here (referenced as
`ADR-NP-XXX`) are the normative source in `numpy-architecture-decisions.md`.

---

## Recently completed

- **Closed the remaining ADR-NP-003 etapa 1 gaps** — the guard layer
  introduced by the previous entry left several confirmed holes, all
  verified by direct execution before and after the fix:
  - A view used **inline** in a list/tuple literal without first being
    bound to a name (`holder = [x[0]]`, `holder = (x[0],)`) escaped the
    container-escape guard entirely (false `VERIFICATION SUCCESSFUL`);
    only the already-named form (`row = x[0]; holder = [row]`) was
    caught. Fixed by recognizing a basic-indexing `Subscript` directly at
    the point of use, gated on the probed result actually being
    array-typed so a scalar element read (`x[0][0]`) packed into a
    container is not misclassified.
  - Storing a view in a **dict literal** (named or inline) crashed the
    solver (`z3_convt::mk_eq` sort-width assertion) instead of producing
    a verdict or diagnostic — `dict_handler_`'s literal-assignment paths
    never reached the container-escape check that already protected
    `List`/`Tuple`. Now rejected explicitly with the same diagnostic.
  - Escape via `global` from a value sourced from a function **parameter**
    (as opposed to a locally created array) is covered by a new
    regression test; already correctly rejected.
  - `a.reshape(d1, d2, ...)` with the dimensions as **separate positional
    arguments** (as opposed to a single tuple) silently dropped every
    dimension past the first. Fixed by collecting all arguments past the
    array as the shape when more than one is given.
  - `a.flatten()`, `a.sum()`, `a.mean()`, `a.min()`, `a.max()`, `a.std()`,
    and `a.var()` — the real NumPy **method forms** — raised an
    unhandled `AttributeError`/`TypeError`; only the `np.<name>(a)`
    module-function forms were recognized. Extended the existing
    method-to-function dispatch rewrite (already used for
    `a.transpose()`/`a.reshape()`/`a.ravel()`) to cover these too, reusing
    each function form's handler unchanged.
  - `a.flat[i] = x` raised the generic `Unsupported assignment target
    type: Call` instead of a diagnostic naming the problem (the
    preprocessor rewrites `.flat` to `np.ravel(a)` even in a write
    target). Now raises a specific `TypeError`; the write itself remains
    unsupported (needs the definitive shared-buffer model, ADR-NP-003
    etapa 2).

  See `regression/numpy/view_inline_*`, `view_dict_literal_escape_fail`,
  `view_global_param_escape_fail`, `reshape_method_multi_arg_*`,
  `flatten_method_*`, `reducer_method_success`, and
  `flat_mutation_diagnostic_fail`.
- **Conservative NumPy view aliasing protection (ADR-NP-003, etapa 1)** —
  basic indexing, row/column slices, strided slices, `transpose`/`.T`,
  `reshape`, and `ravel` are now classified as view-like when they can share
  storage with a source array. Until the definitive shared-buffer descriptor
  model lands, unsafe writes through a copied view, writes to a source array
  while a live copied view exists, view escapes through returns/unknown calls,
  and unsupported identity queries are rejected explicitly instead of producing
  a false `VERIFICATION SUCCESSFUL`. Readonly consumers stay supported:
  `len`, `.shape`, `.ndim`, scalar reads, allowed builtins, and supported
  reducers such as `np.sum(row)`/`np.median(row)` over materialised row views.
  Reassignment/lifetime tracking is conservative across control-flow joins,
  and real copies (`copy`, `flatten`, boolean-mask copies) are kept
  independent. See `regression/numpy/view_*`, `copy_*`, and
  `descriptor_view_*`.
- **Descriptor metadata for classified view copies** — literal slice metadata
  now feeds the canonical `ndarray_descriptor` validation path for view
  shape/rank, so empty views expose a stable logical shape (`shape[0] == 0`,
  `ndim == 1`) instead of carrying a non-constant size expression into
  `.shape`. Non-literal NumPy slice strides are rejected explicitly with
  `TypeError: numpy view slicing requires a literal stride` rather than being
  approximated as step `1`. This is still metadata wiring over the legacy
  copy representation, not the final shared-buffer/offset/stride runtime
  model.
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
  user function with exactly one parameter whose entire body is
  `return <that param>` is inlined at the call site to the caller's own
  argument expression (restricted to a single positional argument, no
  keywords, so no other argument's evaluation or type-checking is skipped),
  since arrays still aren't a valid by-value return type in the current GOTO
  model (see "Missing indexing/slicing" below). This is a narrow,
  call-site-local fix, not a general return mechanism — it does not cover
  `return <param>[index]` or functions with more than one parameter — see
  the note on the general case below. See
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
- **Mixing one literal slice with integer indices in one tuple index**
  (`a[:, 0, 0]`, `a[0, :, 0]`, `a[0:2, 0, 0]`) — an N-D tuple subscript
  with exactly one literal slice axis and every other axis a
  literal/resolvable integer now lowers to a bounded copy along the slice
  axis, generalizing the existing 2-D column-select path. More than one
  slice axis (`a[:, :, 0]`) or symbolic slice bounds stay rejected
  explicitly. See `regression/numpy/numpy_tuple_mixed_slice_*`.
- **Strided slicing (`a[::2]`, `a[1::2]`, `a[::-1]`)** — confirmed already
  supported and now covered by regression tests for 1-D arrays (the
  existing slice model already implemented `step`). Extended to 2-D:
  `a[::2, :]` (strided row selection) and `a[:, ::2]` (strided column
  selection, bare step only — see "Missing indexing / slicing"). `step=0`
  continues to raise `ValueError` at runtime. See
  `regression/numpy/numpy_strided_slice_*`.
- **NumPy API expansion PR** — added or promoted focused support for:
  `np.empty`; `empty_like`/`zeros_like`/`ones_like`/`full_like`;
  concrete `sort`, `argsort`, `searchsorted`, `unique`, `median` and
  `percentile`; scalar and bounded-array `np.random.random`, `rand`,
  `randint`, `uniform`, `choice`, plus explicit `seed` handling; `.flat`
  and readonly `np.nditer`; vector `np.linalg.norm` (`ord` default/2, 1,
  `np.inf`, `-np.inf`) and explicit `solve` size limits; boolean and
  chained `np.dot`; and integer literal `np.transpose` cases that used to
  fall through to an unsafe runtime shape. See the corresponding
  regressions under `regression/numpy/` with prefixes such as `empty_`,
  `like_creation_`, `sort_`, `argsort_`, `searchsorted_`, `unique_`,
  `median_`, `percentile_`, `random_`, `flat`, `nditer`, `norm_`,
  `np_linalg_boundary_`, `dot`, and `transpose`.

---

## Missing indexing / slicing

| Feature | Status | Notes |
|---|---|---|
| Returning a numpy array *out* of a function by value (general case: a sub-array, e.g. `def f(a): return a[0]`, or any non-trivial body) | Missing | Arrays aren't valid by-value return types in the current GOTO model. Only the narrow *identity*-return case is fixed (see "Recently completed") — the general case was attempted twice this round and reverted both times after hitting the same structural wall: **(1)** inlining the substituted return expression at every call site works for the eligibility check but the two-pass assignment machinery (`create_symbol_for_unannotated_assign` type-probes the RHS once, then `get_var_assign` converts it again "for real") evaluates a `Subscript` return expression twice, duplicating the bounds-check GOTO code it emits and corrupting the result (confirmed via `--goto-functions-only`: the second, discarded evaluation's DECLs still land in the block); a cross-call-node cache (keyed by the AST node's address, confirmed to see the same `current_block` on both hits) prevented the double-conversion but did *not* fix the wrong result, meaning the bug is elsewhere in that pipeline. **(2)** A single-member wrapper-struct return type (`struct { value: array_type }`, unwrapped right after a real, once-only function call via the existing `store_call_result` helper — mirroring how a returned tuple already works today) also hit a variant of the same issue: while building it, a separate pre-existing bug was found and fixed (a static Python-level pre-pass injects a wrong `-> Any` return annotation for `return a[0]`, decided *before* parameters are processed, which pre-empted the array-shape detection with a `double` default — now deferred to the post-body GOTO scan, which sees real converted types), but the wrapped struct still isn't reliably unwrapped before a variable's type is decided elsewhere in the same assignment pipeline, causing a segfault (`build_index` dereferencing a struct's `.subtype()`). Both attempts point at the same root cause: `y = f(...)`'s type is decided by more than one code path in `get_var_assign`/`create_symbol_for_unannotated_assign`, not uniformly from what `get_expr` returns. A real fix needs to understand and consolidate that pipeline first, not just work around it at the call site. |
| View aliasing for basic indexing and transpose-like views | Etapa 1 complete | All confirmed etapa-1 guard gaps (inline container escape, dict-literal crash, method-form dispatch) are closed — see "Recently completed". The runtime representation is still a copy, so the final shared-buffer descriptor model (etapa 2) is still missing; that is the only remaining piece of ADR-NP-003. |
| Higher-dimensional or symbolic slice bounds beyond the supported literal-copy cases | Missing | Bounded 2-D column slices with explicit bounds (`a[:, 1:3:2]`) and one- or two-slice-axis mixed tuple indexing (`a[:, 0, 0]`, `a[:, :, 0]`) are supported for literal/fixed-shape cases — confirmed by direct execution (a previous revision of this document listed these as pending, which was stale). Three or more slice axes, symbolic slice bounds, non-literal strides, and broader stride combinations remain rejected explicitly rather than silently approximated. |
| `a.sort()`/`a.argsort()`/`a.tolist()` method forms | Missing | Found while closing the method-dispatch gaps above but not fixed there — different in kind from a plain rewrite: `a.sort()` mutates in place (`np.sort(a)` returns a copy, so redirecting would need `a[:] = np.sort(a)`, not a bare rewrite); `a.argsort()` would redirect to `np.argsort()`, which itself only accepts an inline literal array today, not a variable, so the rewrite wouldn't fix the common case; `a.tolist()` has no module-function equivalent to redirect to at all. |
| `a.any()`/`a.all()` method forms | Missing, cause unclear | `a.any()` raises `ERROR: any() expected at least 1 argument, got 0` — looks like the receiver is being dropped and the call is dispatched to Python's builtin `any()`/`all()` with zero arguments instead of the array. Root cause not investigated; flagged here rather than fixed blind, since it may indicate a broader dispatch issue worth understanding first. |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | Advanced dtype forms (`object`, structured/record dtypes, custom dtype objects) and broad NumPy constructor parity |
| Sorting / searching | Axis-aware and stable-kind variants, `sort`/`argsort` kwargs beyond the supported concrete 1-D recut, and symbolic arrays |
| Statistics | Axis/keepdims/out/overwrite/nan-policy style variants beyond concrete flattened/literal `median` and `percentile` |
| Linear algebra | `inv`/`solve` limited to 2×2/3×3; `norm` limited to vectors and Frobenius matrices; `eig`/`svd` limited to small concrete matrices |
| Random | Additional distributions, full PRNG state semantics, probability-vector `choice`, replacement control, and large/symbolic shapes |
| Structured arrays | Record dtypes |
| Views / strides | Etapa 1 guard model is complete (see "Recently completed"); final shared-buffer alias semantics, writable views, offset/stride based assignment, and broad non-literal stride support (ADR-NP-003 etapa 2) remain missing |
| Iteration | Writable `nditer`, advanced `op_flags`, multi-operand iteration, and mutation through `flat` |

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
4. **Basic-indexing views still do not alias at runtime; covered mutations
   are guarded conservatively** (ADR-NP-003, etapa 1 — complete). NumPy's
   basic indexing (`a[0]`) returns a *view* sharing the source's buffer;
   this frontend still uses a copy representation for the supported
   lowering paths. The direct false-success pattern, the inline-container
   and dict-literal escape gaps, and the missing method-form dispatch are
   all now rejected/supported correctly (see "Recently completed") — the
   remaining piece is the definitive shared-buffer descriptor model
   (etapa 2), which replaces conservative rejection with real aliasing.
5. **`np.dot(a, b)` crashes the process for two 1-D vectors read from a
   variable** (not a literal) — confirmed by direct execution:
   ```python
   import numpy as np
   a = np.array([1, 2, 3])
   b = np.array([4, 5, 6])
   np.dot(a, b)
   ```
   aborts (core dump) during GOTO conversion instead of producing a
   verdict. `a.dot(b)` (method form) does not crash — it raises an
   explicit `TypeError`, so it is merely unsupported, not unsound. Found
   while checking which other methods could safely gain dispatch rewrite
   support alongside this PR; not fixed here since it is unrelated to
   view/dispatch work and needs its own root-cause investigation.

---

## KNOWNBUG tests

No remaining KNOWNBUG in the targeted `dot6`/`dot7`/`transpose2`/
`transpose7` set. `dot6`/`dot7` and `transpose7` are now CORE regressions;
`transpose2` was already CORE and remains covered.

---

## Prioritised next steps

1. **`np.dot(a, b)` crash with two variable-sourced 1-D vectors** — see
   "Soundness concerns" #5. Highest priority: it is a process crash, not
   merely a missing feature or a conservative rejection.
2. **Definitive view descriptor model (ADR-NP-003 etapa 2)** — replace the
   current conservative guard-over-copy approach with shared
   `ndarray_descriptor` buffer_id/offset/stride metadata in indexing,
   assignment, transpose, reshape/ravel, and escape checks. This is the
   path to writable views that alias like NumPy instead of being rejected.
   Etapa 1 (the guard layer) is complete — see "Recently completed".
3. **NumPy arrays as function return values, general case** — only the
   narrow identity-return pattern (a single-parameter function whose entire
   body is `return <that param>`) is fixed; a sub-array return
   (`def f(a): return a[0]`), a function with more than one parameter, or
   any function with more than the one return statement is still
   unsupported. Two implementation strategies were
   tried and reverted this round — see the "Missing indexing / slicing"
   table entry above for the detailed failure analysis. Both hit the same
   root cause: `y = f(...)`'s type is decided by more than one code path
   in the unannotated-assignment machinery
   (`get_var_assign`/`create_symbol_for_unannotated_assign`), not
   uniformly from what `get_expr` returns for the call. A real fix likely
   needs to understand and consolidate that pipeline first, rather than
   work around it purely at the call site or the callee's return
   statement.
4. **`a.sort()`/`a.argsort()`/`a.tolist()`/`a.any()`/`a.all()` method
   forms** — see the "Missing indexing / slicing" table above; each needs
   its own design (in-place mutation semantics, extending `np.argsort()`
   to variables first, a new array-to-list conversion, or root-causing the
   `any()`/`all()` zero-argument dispatch bug) rather than the plain
   rewrite used for the method forms closed in this round.
5. **Multiple slice axes or symbolic bounds mixed with integer indices in
   one tuple** (`a[:, :, :, 0]` on 4-D+, `a[i:j, 0, 0]`) — literal bounded
   one- and two-slice-axis mixed tuple slices (`a[:, 0, 0]`, `a[:, :, 0]`)
   and the bounded strided column slice (`a[:, 1:3:2]`) are already
   supported (confirmed by direct execution — the previous revision of
   this document listed both as still pending, which was stale); broader
   symbolic/multi-axis forms beyond those stay rejected explicitly.
6. **Advanced dtype and constructor parity** — object/structured/custom
   dtype support remains intentionally rejected in the new creation
   helpers; the basic case (`dtype=float` in `np.array`) already works.
7. **Random and iterator follow-up** — extend beyond the initial nondet
   random/choice recut to probability vectors, replacement semantics,
   additional distributions, writable `nditer`, and mutation through `flat`
   (needs etapa 2 first, since `.flat`'s current rewrite target,
   `np.ravel(a)`, is a copy).
8. **Linear algebra breadth** — larger matrices (`det`/`inv` explicitly cap
   at 3×3 today, confirmed by direct execution — generalizing needs a
   symbolic cofactor-expansion/Gauss-elimination algorithm, not a bug fix),
   symbolic matrix entries,
   additional `norm` axes/orders, and more faithful `eig`/`svd` semantics.

## Suggested next PRs

1. **Fix the `np.dot` variable-vector crash** — small, isolated, high
   priority (process crash, not a soundness/feature gap).
2. **Shared-buffer view descriptors (ADR-NP-003 etapa 2)** — connect
   `ndarray_descriptor` buffer/offset/stride metadata to basic indexing,
   transpose, assignment, and escape checks so supported writable views
   alias correctly instead of being rejected conservatively. Also unblocks
   writable `.flat`/`nditer`.
3. **General array returns** — consolidate the assignment/type inference
   pipeline so user functions can return non-trivial NumPy arrays and
   sub-arrays without double conversion or wrapper-type confusion.
4. **Remaining array method forms** — `a.sort()` (in-place semantics),
   `a.argsort()` (needs `np.argsort()` to accept a variable first),
   `a.tolist()` (new conversion, no function-form equivalent), and
   `a.any()`/`a.all()` (root-cause the zero-argument dispatch bug first).
5. **Advanced dtype and constructors** — structured/object dtype policy,
   constructor diagnostics, and dtype propagation for creation/sort/stat
   helpers.
6. **Random and iteration depth** — PRNG-state decisions, remaining
   distributions, probability/replacement `choice`, writable `nditer`, and
   `flat` assignment semantics.
7. **Linear algebra expansion** — matrix-size strategy (`det`/`inv` beyond
   3×3), symbolic-entry policy, and additional `norm`/`eig`/`svd` coverage.

### Out of scope
- True SMT-array scalability — `array_typet` already lowers to SMT
  select/store; see ADR-NP-004. Any further change is benchmark-gated.
- Extending the runtime-list model to hold array-typed elements — rejected
  as an approach for symbolic 2-D boolean-mask row selection (see
  ADR-NP-001's "Alternativas rejeitadas"); still considered
  disproportionately risky for any other use case.
