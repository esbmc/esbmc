# ESBMC NumPy — Remaining Work

**Updated:** 2026-10-12.

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
| Shared-buffer view model | Partial | ADR-NP-003 etapa 3: every supported view is a pointer into its source's storage plus a per-axis shape and stride, which may be run-time values. This covers 1-D to rank 4 slices (literal or symbolic bounds and steps, multi-axis tuples), row/column/subarray views, `transpose`/`.T`/`swapaxes`/`moveaxis`, `reshape`/`ravel`/`.flat` (a view when NumPy's no-copy rule allows, otherwise a copy), `squeeze`, `expand_dims`, read-only `broadcast_to`, `diagonal`, and views of array parameters. Reads and writes, including symbolic indices, go through one address computation, so aliasing holds in both directions. Copies (`np.copy`, `view.copy()`, `np.array(view)`), `tolist()`, flattened reducers, `any`/`all` and basic `nditer` accept these views. A view also keeps its metadata when it leaves direct local use: a call to a function that is a single return expression over its parameters (after any local aliases) is folded into the caller, so views passed in or returned alias as written; a literal `list`/`tuple`/`dict` holding views binds each element to its own view and is read back by literal index or key; `alias = view` carries shape, strides and read-only state; a name bound on two paths to views with the same layout over the same array is one view. Remaining gaps, each rejected with its own `TypeError`: a view passed to a builtin such as `sum`/`list`/`sorted`/`print`/`any`/`all`, a method, a class constructor or a lambda; a view passed to a user function that is not a single return expression when its layout cannot be read from every call site (the source is `np.full`, `np.eye` or `np.identity`, which build lists, or is rebound inside a branch or loop; call sites disagree on the element type or the rank); a view whose layout the callee reads at run time (run-time bounds, or one callee called with different layouts) when the call's arguments contain calls or it sits in a short-circuit operand, a comprehension or a loop condition; a view returned by such a function unless every `return` yields the same parameter; such calls to the right of an array name or attribute in the same expression; value reads or writes of storage after it was passed to a function that does not use it (only `len`/`.shape`/`.ndim`/`.size` stay allowed, there is no havoc); a view stored by `append`, item assignment or a comprehension, in a nested container, an attribute or a global; a container of views that is mutated, indexed by a run-time key, bound inside a branch or loop, or used as a whole; views returned from more than one path; a source name rebound inside a branch or loop while views of it are alive; a name bound on different paths to views with a different rank, element type or read-only state, or to different layouts or arrays inside a loop or while a nested function reads the name; an array rebound while a view bound per path to it or to another array is alive; advanced `nditer`; `reshape`/`ravel` of an N-D view whose stride is only known at run time (whether NumPy copies depends on that stride); an N-D slice with run-time bounds that is used without being bound to a name. |

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
| Views / strides | Views passed to builtins, methods, constructors or lambdas; views passed to multi-statement functions outside the subset whose layout is read from the call sites (see the shared-buffer view model row); views stored by mutation, in attributes or globals; `reshape`/`ravel` of N-D views with a run-time stride; arrays whose shape has no static upper bound (`np.zeros(n)` with `n` unconstrained). |
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
6. **A call in a right operand runs before the left operand is read.** In
   `x = g + bump()`, where `bump()` assigns the global `g`, the Python
   frontend reads `g` after the call; Python reads it before. This is a
   frontend-wide wrong success, not NumPy-specific. A call that passes a
   NumPy view to a user function is exempt: the operands to its left are
   read first (`view_param_operand_order_success`).
7. **A reduction on a row of a strided 2-D view gives a false alarm**:
   `v = a[1:]; v[1].sum()` fails although the value is correct; `v.sum()` and
   element reads are right.
8. **Writing a float into an integer array ends in a solver exception**
   instead of truncating (`a = np.array([1, 2]); a[0] = 2.7`), with no
   verdict. `np.arange` bound to a name is such an array.
9. **A local first bound inside a branch of a function does not outlive
   the branch.** `if c: b = [1, 2]` followed by `b[0]` gives a false alarm
   for any pointer-typed local. NumPy view names are declared at the top
   of the function and are not affected; lists and other objects are.
10. **An array name rebound to a view of itself gives a false alarm**:
    `x = np.zeros((4, 4)); y = x[1]; x = y` retypes `x` and breaks `y`.
    `np.eye` and `np.identity` stay lists for this reason
    (`shape_cache_view_rebind_success`); `np.arange` and `np.linspace`
    bound to a name are arrays and share it.
11. **A generator over a NumPy array inside `sum()` reads wrong values**
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

The one known wrong success is the operand evaluation order of the Python
frontend ("Soundness / performance concerns", item 6), which is not specific
to NumPy. Every other remaining item is documented backlog: unsupported
cases should reject explicitly, and known false alarms are listed as gaps
instead of treated as supported behavior.

`regression/numpy` pins eight `KNOWNBUG` tests (2026-10-12):

| test | expected | today |
|---|---|---|
| `e`, `round2` | SUCCESSFUL | `VERIFICATION FAILED` (false alarm) |
| `view_rebind_sibling_name_alias_knownbug` | SUCCESSFUL | `VERIFICATION FAILED` (false alarm) |
| `isclose`, `nextafter`, `remainder` | SUCCESSFUL | internal error (an expression dump, no verdict) |
| `array_return_call_arg_edge` | SUCCESSFUL | rejected: `'int' object is not subscriptable` |
| `det2` | FAILED | rejected: `numpy.linalg.det supports only 2x2 and 3x3 matrices` |

A view passed to a multi-statement user function keeps its shape, strides
and aliasing: subviews taken in the callee, nested and recursive callees,
row slices of a 2-D array, views with run-time bounds or with a different
layout per call site, view expressions and view-returning calls inside a
larger expression (operands to their left are read first, as in Python),
callees defined after their caller, and sources that are
an alias, `np.zeros`/`np.ones`/`np.arange`/`np.linspace`, or a function's
literal array. A view name bound on
different paths to views with different layouts or of different arrays
keeps the metadata of the path taken. The cases still outside that subset
reject, each pinned by a `CORE` test:

| test | NumPy | ESBMC |
|---|---|---|
| `view_param_builtin_remaining_fail`, `view_param_method_remaining_fail`, `view_param_constructor_remaining_fail`, `view_param_lambda_remaining_fail` | a builtin, method, class constructor or lambda receives the view | rejected: only plain user functions bind a view parameter |
| `view_param_unknown_source_remaining_fail` | a view of `np.full(...)`, `np.eye(...)` or `np.identity(...)` aliases it | rejected: these build a list whose slices are copies, as do `np.arange` and `np.linspace` called with a keyword |
| `view_param_element_type_conflict_fail` | one function takes an int view and a float view | rejected: view passed to a multi-statement function |
| `view_param_branch_source_rebind_fail` | the source array is rebound inside a branch before the view is taken | rejected: view passed to a multi-statement function |
| `view_param_operand_array_name_fail` | `b + f(v)[0]` with an array name `b` to the left of a view-returning call | rejected: an array operand is not bound to a temporary, which a later call could change through an alias |
| `view_param_return_multi_path_fail` | a function returns one of two view parameters | rejected: returning a copied numpy view |
| `view_branch_merge_loop_fail` | a view name rebound to another layout inside a loop | rejected: different shape, strides or storage on different paths |
| `view_branch_merge_nested_function_fail` | a view name rebound on a path while a nested function reads it | rejected: different shape, strides or storage on different paths |
| `view_branch_element_type_fail` | a name bound per branch to an int view and a float view | rejected: different shape, strides or storage on different paths |
| `view_branch_storage_rebind_fail` | an array is rebound while a view bound per branch to it or to another array is alive | rejected: which storage the view points at is only known at run time |

---

## Prioritised next steps

No item is a known soundness gap; the backlog, in priority order:

1. **Views in general callees** — builtins, methods, class constructors and
   lambdas receiving a view. The residual forms in the table above are
   narrow and each needs something outside the view model: a per-type
   copy of the callee, or `np.full`, `np.eye` and `np.identity` building
   arrays.
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

1. **Views in general callees** (~1 PR) — views passed to builtins,
   methods, constructors and lambdas.
2. **Advanced dtype and constructors** (~2 PRs) — dtype policy
   (object/structured/custom) separate from constructor
   diagnostics/propagation.
3. **Random and iteration depth** (~2 PRs) — new distributions/`choice`
   separate from advanced `nditer`.
4. **Linear algebra expansion** (~2 PRs) — larger/symbolic matrix support
   separate from fuller `eig`/`svd`/`norm`.

**Total to close every item in this file: ~7 PRs.** The operand evaluation
order and the branch-declared locals ("Soundness / performance concerns",
items 6 and 9) are Python frontend fixes outside this count.

---

## Out of scope

- True SMT-array scalability beyond the current `array_typet` lowering; see
  ADR-NP-004.
- Extending the runtime-list model to hold array-typed elements; this remains
  disproportionately risky for current NumPy goals.
