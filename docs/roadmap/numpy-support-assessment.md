# ESBMC NumPy — Remaining Work

**Updated:** 2026-10-09.

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
| Shared-buffer view model | Partial | ADR-NP-003 etapa 3: every supported view is a pointer into its source's storage plus a per-axis shape and stride, which may be run-time values. This covers 1-D to rank 4 slices (literal or symbolic bounds and steps, multi-axis tuples), row/column/subarray views, `transpose`/`.T`/`swapaxes`/`moveaxis`, `reshape`/`ravel`/`.flat` (a view when NumPy's no-copy rule allows, otherwise a copy), `squeeze`, `expand_dims`, read-only `broadcast_to`, `diagonal`, and views of array parameters. Reads and writes, including symbolic indices, go through one address computation, so aliasing holds in both directions. Copies (`np.copy`, `view.copy()`, `np.array(view)`), `tolist()`, flattened reducers, `any`/`all` and basic `nditer` accept these views. A view also keeps its metadata when it leaves direct local use: a call to a function that is a single return expression over its parameters (after any local aliases) is folded into the caller, so views passed in or returned alias as written; a literal `list`/`tuple`/`dict` holding views binds each element to its own view and is read back by literal index or key; `alias = view` carries shape, strides and read-only state; a name bound on two paths to views with the same layout over the same array is one view. Remaining gaps, each rejected with its own `TypeError`: a view passed to a builtin such as `sum`/`list`/`sorted`/`print`/`any`/`all`, a method, a class constructor or a lambda; a view passed to a user function that is not a single return expression when its layout cannot be read from every call site (the source is not a literal `np.array` bound once in the scope of the call, a slice bound is not a literal, the argument is a slice of a 2-D array, call sites disagree on the layout, the callee is defined after its caller, or the argument is a view expression anywhere but directly in `x = f(...)` or `f(...)`); inside such a function, a write through a subview of the parameter, a subview passed on to another function, and a call that passes the parameter on while a copied subview of it is live; a view returned by such a function unless every `return` yields the same parameter and the call is the whole right-hand side of an assignment; value reads or writes of storage after it was passed to a function that does not use it (only `len`/`.shape`/`.ndim`/`.size` stay allowed, there is no havoc); a view stored by `append`, item assignment or a comprehension, in a nested container, an attribute or a global; a container of views that is mutated, indexed by a run-time key, bound inside a branch or loop, or used as a whole; views returned from more than one path; a source name rebound inside a branch or loop while views of it are alive; a name bound on different paths to views with a different shape, strides or storage; advanced `nditer`; `reshape`/`ravel` of an N-D view whose stride is only known at run time (whether NumPy copies depends on that stride); an N-D slice with run-time bounds that is used without being bound to a name. |

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
| Views / strides | Views passed to builtins, methods, constructors or lambdas; views passed to multi-statement functions outside the subset whose layout is read from the call sites, and subviews written or passed on inside such a function (see the shared-buffer view model row); views stored by mutation, in attributes or globals; path-sensitive view metadata for branches with different layouts; `reshape`/`ravel` of N-D views with a run-time stride; arrays whose shape has no static upper bound (`np.zeros(n)` with `n` unconstrained). |
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
5. **A second name for a rebound array does not follow its old storage.**
   After `b = a` and `a = np.array(...)`, a view of the old `a` writes to the
   old storage while `b` reads a separate copy of it, so a write through the
   view is not seen through `b` (`view_rebind_sibling_name_alias_knownbug`).
   NumPy keeps `b` on the old storage. This is a false alarm, not a wrong
   success.
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

This file lists **no known NumPy unsound-success gap**. Every remaining item
is documented backlog: unsupported cases should reject explicitly, and known
false alarms are listed as gaps instead of treated as supported behavior.

`regression/numpy` pins nine `KNOWNBUG` tests (2026-10-09):

| test | expected | today |
|---|---|---|
| `e`, `round2` | SUCCESSFUL | `VERIFICATION FAILED` (false alarm) |
| `view_branch_registration_conflict_knownbug` | SUCCESSFUL | rejected: binding a view name to views with a different shape on different paths (needs path-sensitive view metadata) |
| `view_rebind_sibling_name_alias_knownbug` | SUCCESSFUL | `VERIFICATION FAILED` (false alarm) |
| `isclose`, `nextafter`, `remainder` | SUCCESSFUL | internal error (an expression dump, no verdict) |
| `array_return_call_arg_edge` | SUCCESSFUL | rejected: `'int' object is not subscriptable` |
| `det2` | FAILED | rejected: `numpy.linalg.det supports only 2x2 and 3x3 matrices` |

View-parameter cases outside the supported subset reject, each pinned by a
`CORE` test:

| test | NumPy | ESBMC |
|---|---|---|
| `view_param_subslice_write_fail` | a write through `sub = row[1:]` reaches the source | rejected: writing through a copied numpy view |
| `view_param_nested_subview_call_fail` | a subview passed on aliases the source | rejected: view passed to a multi-statement function |
| `view_param_live_copy_nested_call_fail` | a subview sees a write made by a nested callee | rejected: copied view of the storage is live |
| `view_param_direct_expr_nested_fail` | `f(a[0:2], 7) + 1` writes into `a` | rejected: view expression not bound to a name |
| `view_param_unanalyzable_site_fail` | a view with a run-time bound (`a[0:n]`) is passed like any other | rejected: view passed to a multi-statement function |
| `view_param_2d_slice_fail` | a 2-D slice is read and written in the callee | rejected: view passed to a multi-statement function |
| `view_param_callee_order_fail` | the callee may be defined after its caller | rejected: view passed to a multi-statement function |
| `view_param_branch_storage_conflict_fail` | a name bound per branch to views of two arrays | rejected: different storage on different paths |
| `view_param_return_view_subscript_fail` | `f(v)[0]` reads through the returned view | rejected: returning a copied numpy view |

---

## Prioritised next steps

No item is a known soundness gap; the backlog, in priority order:

1. **Views in general callees** — the view-parameter cases in the table above
   (subviews of a parameter as real views, 2-D slices, views with run-time
   bounds, view expressions nested in a larger expression, returned views
   used as values, callees defined after their caller, sources that are not
   literals bound once); then builtins,
   methods, constructors and lambdas.
2. **Advanced dtype and constructor parity** — structured/object/custom dtype
   policy, diagnostics, and propagation.
3. **Random and iteration depth** — probability/replacement `choice`, extra
   distributions, and advanced `nditer`.
4. **Linear algebra breadth** — larger matrices, symbolic entries, and more
   faithful `norm`/`eig`/`svd`.

---

## Suggested next PRs

Each roadmap item above groups several sub-efforts; sizing them 1 PR per
item undercounts the real work. Items below with multiple named consumers or
distinct designs are sized accordingly instead of assumed to be one PR each.

1. **Views in general callees** (~2 PRs) — the remaining view-parameter
   forms listed under "Community testing readiness", separate from
   builtins/methods/constructors/lambdas and path-sensitive metadata for
   branches.
2. **Advanced dtype and constructors** (~2 PRs) — dtype policy
   (object/structured/custom) separate from constructor
   diagnostics/propagation.
3. **Random and iteration depth** (~2 PRs) — new distributions/`choice`
   separate from advanced `nditer`.
4. **Linear algebra expansion** (~2 PRs) — larger/symbolic matrix support
   separate from fuller `eig`/`svd`/`norm`.

**Total to close every item in this file: ~8 PRs.**

---

## Out of scope

- True SMT-array scalability beyond the current `array_typet` lowering; see
  ADR-NP-004.
- Extending the runtime-list model to hold array-typed elements; this remains
  disproportionately risky for current NumPy goals.
