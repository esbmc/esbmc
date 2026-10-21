# ESBMC NumPy — Remaining Work

**Updated:** 2026-10-19.

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
| Shared-buffer view model | Partial | ADR-NP-003 etapa 3: every supported view is a pointer into its source's storage plus a per-axis shape and stride, which may be run-time values. This covers 1-D to rank 4 slices (literal or symbolic bounds and steps, multi-axis tuples), row/column/subarray views, `transpose`/`.T`/`swapaxes`/`moveaxis`, `reshape`/`ravel`/`.flat` (a view when NumPy's no-copy rule allows, otherwise a copy), `squeeze`, `expand_dims`, read-only `broadcast_to`, `diagonal`, and views of array parameters. Reads and writes, including symbolic indices, go through one address computation, so aliasing holds in both directions. Copies (`np.copy`, `view.copy()`, `np.array(view)`), `tolist()`, flattened reducers, `any`/`all` and basic `nditer` accept these views. A view also keeps its metadata when it leaves direct local use: a call to a function that is a single return expression over its parameters (after any local aliases) is folded into the caller, so views passed in or returned alias as written; a literal `list`/`tuple`/`dict` holding views binds each element to its own view and is read back by literal index or key; `alias = view` carries shape, strides and read-only state; a name bound on two paths to views with the same layout over the same array is one view. A 1-D view passed to `sum`, `any`, `all`, `min`, `max`, `tuple`, `reversed`, `print`, `list` or `sorted` is read as `view.tolist()`, `sorted`/`min`/`max` with `key=` (and `reverse=` or `default=`, constant or run-time) included, and a copied view (of a list-backed source) is read as it is by the same builtins except `list`; a view passed to an ndarray method, a user method, a class constructor or a module-level lambda over its parameters binds like a function parameter. Remaining gaps, each rejected with its own `TypeError`: a view passed to `list()` when it is a copy, to a lambda that is passed on as a callback, through `**kwargs`, or to a function of another non-NumPy module; a view passed to a user function that is not a single return expression when its layout cannot be read from every call site (the source is `np.full` with an unbounded size or with an `order=`/`device=`/`like=` keyword, or is rebound inside a branch or loop; call sites disagree on the element type or the rank); a view whose layout the callee reads at run time (run-time bounds, or one callee called with different layouts) when the call's arguments contain calls or it sits in a short-circuit operand, a comprehension or a loop condition; a view returned by such a function unless every `return` yields the same parameter; such calls to the right of an array name or attribute in the same expression; value reads or writes of storage after it was passed to a function that does not use it (only `len`/`.shape`/`.ndim`/`.size` stay allowed, there is no havoc); a view stored by `append`, item assignment or a comprehension, in a nested container, an attribute or a global; a container of views that is mutated, indexed by a run-time key, bound inside a branch or loop, or used as a whole; views returned from more than one path; a source name rebound inside a branch or loop while views of it are alive; a name bound on different paths to views with a different rank, element type or read-only state, or to different layouts or arrays inside a loop or while a nested function reads the name; an array rebound while a view bound per path to it or to another array is alive; advanced `nditer`; `reshape`/`ravel` of an N-D view whose stride is only known at run time (whether NumPy copies depends on that stride); an N-D slice with run-time bounds that is used without being bound to a name. |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | `object`, structured/record and custom dtype objects (`np.dtype(...)`), dtype strings with a width or a type code (`"U3"`, `"i4"`, `"V8"`) and a dtype held in a variable reject explicitly. Integer and float dtypes narrower than 64 bits are applied as the constructor's cast (truncation, `OverflowError` for an out-of-range integer, NumPy's `arange` wraparound) but stored at 64 bits, so later arithmetic does not wrap or round at the dtype's width (an `int8` array would read as a string, and NumPy compares a `float32` element with a Python float in `float32`). String arrays are fixed-width: element reads, writes (truncated to the width), comparison, `len`, copies and iteration are modelled; slicing, `in` and passing them to a function reject. Complex arrays support element reads, writes, comparison, `.real`/`.imag`, copies, slices and iteration; reducers, arithmetic, builtins and passing them to a function reject. `np.full` with an array fill (broadcasting) or with `dtype=` and a non-literal fill, a number converted to a string dtype, `np.linspace` with `axis=`/`device=`, `retstep=True` not unpacked into two names, or a non-constant `start`/`stop`/`num`, and `arange`/`linspace` above 10000 elements reject. A constructor imported by name (`from numpy import full`) is not recognised. Broad constructor parity beyond `zeros`/`ones`/`full`/`array`/`eye`/`identity`/`linspace`/`arange`. |
| Element-wise operators | `+ - * / // ** & \| ^ << >>` are modelled between a numeric array and a number on either side or an array of the same shape, with NumPy's result dtype (float64 for `/` and for any float operand; integer `//` and `%` by zero give 0); a bitwise operator with a float operand raises NumPy's `TypeError`. Arrays of different shapes (broadcasting), `%` other than `int array % int`, and an integer array raised to a non-constant or negative power reject explicitly. |
| Sorting / searching | Symbolic array shapes and symbolic `sorter=` operands for `searchsorted` reject explicitly. Genuine 2-D array input is also rejected, matching NumPy's `ValueError: object too deep for desired array`; 1-D row/column views remain supported. |
| Direct constructor chaining | Supported for the current concrete constructor subset and direct literal `np.array(...)` receivers across the pinned reducers/views (`sum`, `mean`, `min`, `max`, `argsort`, `searchsorted`, `flatten`, `ravel`, `transpose`, `.T`, `reshape`, `argmin`, `argmax`, `diagonal`, `prod`, `std`, and `var`). Remaining gaps are method forms tied to unsupported constructors, symbolic constructor payloads, or broader unsupported APIs. |
| Statistics | `a.sum()`/`a.mean()`/`a.min()`/`a.max()`/`a.any()`/`a.all()`/`a.argmin()`/`a.argmax()` method forms and their `axis=0/1` variants are supported over concrete 1-D/2-D ndarrays (including function-returned arrays), sharing the same reducer/comparison policy as the functional forms. Still missing: axis/keepdims/out/overwrite/nan-policy style variants beyond concrete flattened/literal `median` and `percentile`, and reducer axes outside 2-D concrete `axis=0/1`. |
| Linear algebra | `det`/`inv`/`solve` beyond small concrete matrices, symbolic matrix entries, additional `norm` axes/orders, and fuller `eig`/`svd` semantics. |
| Random | Generator state and objects (`seed`, `get_state`, `set_state`, `default_rng`, `Generator`, `RandomState`) and every unmodelled `np.random` function (`shuffle`, `permutation`, `multivariate_normal`, `laplace`, `logistic`, ...) raise `TypeError`; `choice(p=...)` does too, since a probability vector has no encoding that keeps ADR-NP-002's over-approximation. Parameters must be scalars (no broadcasting of array parameters). `size` must be integer literals, at most 1024 samples (each sample is unrolled); a `size` read at run time rejects. `choice` takes a 1-D literal sequence, ndarray or view, not an integer `a` or a slice of a string array. `randint` rejects `dtype=bool` and `uint64`, and narrow integer dtypes are stored at 64 bit as for the constructors. |
| Structured arrays | Record dtypes. |
| Views / strides | Views passed to lambdas used as callbacks, through `**kwargs` or to other modules' functions; views passed to multi-statement functions outside the subset whose layout is read from the call sites (see the shared-buffer view model row); views stored by mutation, in attributes or globals; `reshape`/`ravel` of N-D views with a run-time stride; arrays whose shape has no static upper bound (`np.zeros(n)` with `n` unconstrained). |
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
7. **A NumPy array rebound to another shape or dtype** gets its own symbol
   only in straight-line code that no other function reads; inside a
   branch or loop, or while a function reads the name, the rebind rejects
   with a `TypeError` (`array_rebind_other_shape_*_fail`).
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

`regression/numpy` pins eight `KNOWNBUG` tests (2026-10-15):

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
callees defined after their caller, methods, class constructors,
module-level lambdas, and sources that are an alias,
`np.zeros`/`np.ones`/`np.full`/`np.eye`/`np.identity`/`np.arange`/`np.linspace`,
or a function's literal array. A view name bound on
different paths to views with different layouts or of different arrays
keeps the metadata of the path taken. The cases still outside that subset
reject, each pinned by a `CORE` test:

| test | NumPy | ESBMC |
|---|---|---|
| `view_general_lambda_escape_fail` | a lambda that receives the view stores it elsewhere | rejected: the lambda is not a plain function |
| `view_general_external_method_fail` | a function of another module receives the view | rejected: function of a non-NumPy module |
| `view_param_element_type_conflict_fail` | one function takes an int view and a float view | rejected: view passed to a multi-statement function |
| `view_param_branch_source_rebind_fail` | the source array is rebound inside a branch before the view is taken | rejected: rebinding a numpy array to another shape inside a branch |
| `view_param_operand_array_name_fail` | `b + f(v)[0]` with an array name `b` to the left of a view-returning call | rejected: an array operand is not bound to a temporary, which a later call could change through an alias |
| `view_param_return_multi_path_fail` | a function returns one of two view parameters | rejected: returning a copied numpy view |
| `view_branch_merge_loop_fail` | a view name rebound to another layout inside a loop | rejected: different shape, strides or storage on different paths |
| `view_branch_merge_nested_function_fail` | a view name rebound on a path while a nested function reads it | rejected: different shape, strides or storage on different paths |
| `view_branch_element_type_fail` | a name bound per branch to an int view and a float view | rejected: different shape, strides or storage on different paths |
| `view_branch_storage_rebind_fail` | an array is rebound while a view bound per branch to it or to another array is alive | rejected: which storage the view points at is only known at run time |

---

## Prioritised next steps

No item is a known soundness gap; the backlog, in priority order:

1. **Iteration depth** — advanced `nditer` (flags, multiple operands,
   `multi_index`, `external_loop`, buffering, `op_dtypes`/`op_axes`, writable
   items).
2. **Linear algebra breadth** — larger matrices, symbolic entries, and more
   faithful `norm`/`eig`/`svd`.
3. **Dtype width and non-numeric arrays** — storage and arithmetic in the
   dtype's own width (NEP 50 promotion), and the string/complex array
   consumers that reject today; `object`/structured/custom dtypes stay out.
4. **Random generator state** — `Generator`/`default_rng`, `seed` and
   probability-vector `choice`; each needs its own ADR-NP-002 decision first.

The residual view forms in the table above (lambdas passed on, `**kwargs`, a per-type copy of a callee) are narrow and each
needs something outside the view model; they stay rejected.

---

## Suggested next PRs

The backlog closes in four PRs:

1. **Advanced `nditer` and rebinding** — the iteration depth above, plus a
   second name that follows a rebound array's old storage (item 5) and a
   rebind to another shape inside a branch or loop, or while a function
   reads the name (item 7).
2. **Linear algebra expansion** — larger and symbolic matrices together
   with fuller `eig`/`svd`/`norm`; the largest of the four.
3. **Dtype width and non-numeric arrays** — narrow-width storage and
   arithmetic, string/complex array consumers.
4. **Random generator state** — after the ADR decision on state and
   probabilities.

The operand evaluation order (item 6) is a Python frontend fix outside
these four.

---

## Out of scope

- True SMT-array scalability beyond the current `array_typet` lowering; see
  ADR-NP-004.
- Extending the runtime-list model to hold array-typed elements; this remains
  disproportionately risky for current NumPy goals.
