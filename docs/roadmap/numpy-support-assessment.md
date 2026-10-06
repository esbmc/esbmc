# ESBMC NumPy — Remaining Work

**Updated:** 2026-10-05.

This file tracks only what is **not yet implemented, broken, risky, or queued
as backlog** in the NumPy module. If an item is not listed here as a gap, TODO,
or backlog entry, treat it as already implemented and covered by git history
and `regression/numpy/`.

Architectural decisions that gate specific pendencies here are referenced as
`ADR-NP-XXX`. The file meant to hold them, `numpy-architecture-decisions.md`,
was never committed, so the only definitions are the uses in this document and
in `src/python-frontend/` comments: ADR-NP-003 is the shared-buffer view model
and its stages, ADR-NP-004 the SMT-array scalability decision, and "principle
3" the rule that an unsupported case rejects with a diagnostic.

---

## Missing indexing / slicing

| Feature | Status | Notes |
|---|---|---|
| General NumPy array returns from user functions | Partial | Concrete array/view/descriptor returns from supported constructors, bare parameters, parameter subscript/subarrays, and supported descriptor calls over parameters are implemented. Remaining gap: `array_return_call_arg_edge` is still pinned as `KNOWNBUG`; mutating a captured list/array inside a function without a `global` declaration is a pre-existing symbol-resolution gap, not specific to array returns. Genuinely symbolic parameter shapes are rejected when unsupported metadata/view consumers need concrete shape information; `len()` on such parameters remains supported through its independent path. |
| Shared-buffer view model | Partial | ADR-NP-003 etapa 3: every supported view is a pointer into its source's storage plus a per-axis shape and stride, which may be run-time values. This covers 1-D to rank 4 slices (literal or symbolic bounds and steps, multi-axis tuples), row/column/subarray views, `transpose`/`.T`/`swapaxes`/`moveaxis`, `reshape`/`ravel`/`.flat` (a view when NumPy's no-copy rule allows, otherwise a copy), `squeeze`, `expand_dims`, read-only `broadcast_to`, `diagonal`, and views of array parameters. Reads and writes, including symbolic indices, go through one address computation, so aliasing holds in both directions. Copies (`np.copy`, `view.copy()`, `np.array(view)`), `tolist()`, flattened reducers, `any`/`all` and basic `nditer` accept these views. A view also keeps its metadata when it leaves direct local use: a call to a function that is a single return expression over its parameters (after any local aliases) is folded into the caller, so views passed in or returned alias as written; a literal `list`/`tuple`/`dict` holding views binds each element to its own view and is read back by literal index or key; `alias = view` carries shape, strides and read-only state; a name bound on two paths to views with the same layout over the same array is one view. Remaining gaps, each rejected with its own `TypeError`: a view passed to any other callee (a function that uses it outside a single return expression, reads a global, or binds a call to a local that is unused, rebound or would be reordered by inlining, a builtin such as `sum`/`list`/`sorted`/`print`, a method, a class constructor, a lambda); value reads or writes of storage after it was passed to a function that does not use it (only `len`/`.shape`/`.ndim`/`.size` stay allowed, there is no havoc); a view stored by `append`, item assignment or a comprehension, in a nested container, an attribute or a global; a container of views that is mutated, indexed by a run-time key, bound inside a branch or loop, or used as a whole; views returned from more than one path; a name bound on different paths to views with a different shape, strides or storage; advanced `nditer`; `reshape`/`ravel` of an N-D view whose stride is only known at run time (whether NumPy copies depends on that stride); an N-D slice with run-time bounds that is used without being bound to a name. |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | Advanced dtype forms (`object`, structured/record dtypes, custom dtype objects) still reject explicitly; broad constructor parity beyond `zeros`/`ones`/`full`/`array`/`eye`/`identity`/`linspace`/`arange`. |
| Sorting / searching | Symbolic array shapes and symbolic `sorter=` operands for `searchsorted` reject explicitly. Genuine 2-D array input is also rejected, matching NumPy's `ValueError: object too deep for desired array`; 1-D row/column views remain supported. |
| Direct constructor chaining | Supported for the current concrete constructor subset and direct literal `np.array(...)` receivers across the pinned reducers/views (`sum`, `mean`, `min`, `max`, `argsort`, `searchsorted`, `flatten`, `ravel`, `transpose`, `.T`, `reshape`, `argmin`, `argmax`, `diagonal`, `prod`, `std`, and `var`). Remaining gaps are method forms tied to unsupported constructors, symbolic constructor payloads, or broader unsupported APIs. |
| Statistics | `a.sum()`/`a.mean()`/`a.min()`/`a.max()`/`a.any()`/`a.all()`/`a.argmin()`/`a.argmax()` method forms and their `axis=0/1` variants are supported over concrete 1-D/2-D ndarrays (including function-returned arrays), sharing the same reducer/comparison policy as the functional forms. Still missing: axis/keepdims/out/overwrite/nan-policy style variants beyond concrete flattened/literal `median` and `percentile`, and reducer axes outside 2-D concrete `axis=0/1`. |
| Linear algebra | `det`/`inv`/`solve` beyond small concrete matrices, symbolic matrix entries, additional `norm` axes/orders, and fuller `eig`/`svd` semantics. |
| Random | Additional distributions, full PRNG state semantics, probability-vector `choice`, replacement control, and large/symbolic shapes. |
| Structured arrays | Record dtypes. |
| Views / strides | Views passed to functions that are not a single return expression, to builtins, methods or constructors; views stored by mutation, in attributes or globals; path-sensitive view metadata for branches with different layouts; `reshape`/`ravel` of N-D views with a run-time stride; arrays whose shape has no static upper bound (`np.zeros(n)` with `n` unconstrained). |
| Iteration | Advanced `nditer` flags/options, multi-operand iteration, `external_loop`, `multi_index`, buffering, non-C order, casting/op_dtypes/op_axes, and broader mutable item forms. |

---

## Soundness / performance concerns

1. **Constant-folding bypasses ESBMC's overflow/rounding checks** for folded
   paths. Use `--python-no-fold` to force SMT encoding and compare verdicts.
2. **Element-wise broadcasting** still requires concrete shapes at conversion
   time; symbolic shapes work only for selected array creation paths.
3. **Scalability wall** (#5121, closed): arrays are still represented as fully
   unrolled value lists. Large arrays can explode even when the operation is
   conceptually simple.
4. **Views alias through pointers, not through copied mirrors.** A symbolic
   index on an array or a view is normalized like NumPy's (negative values
   count from the end) and an out-of-range one raises a catchable
   `IndexError`; before, a negative symbolic index on a plain array was read
   unnormalized. Arrays with no static bound on their shape remain out of
   scope.
5. **Sibling views stop aliasing after their source name is rebound.**
   Rebinding the source detaches each live view into its own snapshot, so a
   write through one view of the old array is no longer seen through
   another (`view_rebind_sibling_views_alias_knownbug`). NumPy keeps both on
   the old storage. This is a wrong `VERIFICATION SUCCESSFUL`, not a
   rejection.
6. **A generator over a NumPy array inside `sum()` reads wrong values**
   (`sum(v for v in a)` gives a false alarm; the plain `for` loop is
   correct), and `np.transpose` of an array with a run-time shape fails with
   a raw JSON exception instead of a diagnostic.
---

## Community testing readiness

ESBMC's standard across every frontend (C, C++, Solidity, Java/Kotlin) is
sound-but-incomplete, not full language/library coverage: whatever falls
outside the currently supported subset must reject with an explicit
diagnostic (ADR-NP principle 3) rather than silently return a wrong
verdict. Gaps in "Missing indexing / slicing" and "Missing API
surface" should reject explicitly; any known false alarm must be listed as a
separate gap instead of treated as supported behavior.

This file lists **one known NumPy unsound-success gap**: sibling views after
a source rebind (item 5 above). Every other remaining item is documented
backlog: unsupported cases should reject explicitly, and known false alarms
are listed as gaps instead of treated as supported behavior.

`regression/numpy` pins nine `KNOWNBUG` tests (2026-10-05):

| test | expected | today |
|---|---|---|
| `e`, `round2` | SUCCESSFUL | `VERIFICATION FAILED` (false alarm) |
| `view_branch_registration_conflict_knownbug` | SUCCESSFUL | rejected: binding a view name to views with a different shape on different paths (needs path-sensitive view metadata) |
| `view_rebind_sibling_views_alias_knownbug` | FAILED | `VERIFICATION SUCCESSFUL` (unsound) |
| `isclose`, `nextafter`, `remainder` | SUCCESSFUL | internal error (an expression dump, no verdict) |
| `array_return_call_arg_edge` | SUCCESSFUL | rejected: `'int' object is not subscriptable` |
| `det2` | FAILED | rejected: `numpy.linalg.det supports only 2x2 and 3x3 matrices` |

---

## Prioritised next steps

Item 1 is a soundness fix; the rest is backlog, in priority order:

1. **Sibling views after a source rebind** — detach every view of the
   rebound array onto one shared snapshot, so they keep aliasing each other
   (the unsound-success gap above).
2. **Views in general callees** — give a view parameter its shape and strides
   inside a function that is not a single return expression, so its body can
   read and write the view; then builtins, methods and constructors.
3. **Advanced dtype and constructor parity** — structured/object/custom dtype
   policy, diagnostics, and propagation.
4. **Random and iteration depth** — probability/replacement `choice`, extra
   distributions, and advanced `nditer`.
5. **Linear algebra breadth** — larger matrices, symbolic entries, and more
   faithful `norm`/`eig`/`svd`.

---

## Suggested next PRs

Each roadmap item above groups several sub-efforts; sizing them 1 PR per
item undercounts the real work. Items below with multiple named consumers or
distinct designs are sized accordingly instead of assumed to be one PR each.

1. **Sibling views after a source rebind** (~1 PR) — one shared snapshot per
   rebound array.
2. **Views in general callees** (~2 PRs) — typed view parameters for
   multi-statement functions, separate from builtins/methods/constructors and
   path-sensitive metadata for branches.
3. **Advanced dtype and constructors** (~2 PRs) — dtype policy
   (object/structured/custom) separate from constructor
   diagnostics/propagation.
4. **Random and iteration depth** (~2 PRs) — new distributions/`choice`
   separate from advanced `nditer`.
5. **Linear algebra expansion** (~2 PRs) — larger/symbolic matrix support
   separate from fuller `eig`/`svd`/`norm`.

**Total to close every item in this file: ~9 PRs.**

---

## Out of scope

- True SMT-array scalability beyond the current `array_typet` lowering; see
  ADR-NP-004.
- Extending the runtime-list model to hold array-typed elements; this remains
  disproportionately risky for current NumPy goals.
