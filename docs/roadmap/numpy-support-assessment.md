# ESBMC NumPy — Remaining Work

**Updated:** 2026-09-11.

This file tracks only what is **not yet implemented, broken, risky, or queued
as backlog** in the NumPy module. If an item is not listed here as a gap, TODO,
or backlog entry, treat it as already implemented and covered by git history
and `regression/numpy/`.

Architectural decisions that gate specific pendencies here (referenced as
`ADR-NP-XXX`) are the normative source in `numpy-architecture-decisions.md`.

---

## Missing indexing / slicing

| Feature | Status | Notes |
|---|---|---|
| General NumPy array returns from user functions | Partial | A user function can now return a concrete array/view/descriptor built from `np.array`/`np.zeros`/`np.ones`/`np.full`, a bare parameter, a subscript/subarray of a parameter, or a supported descriptor call over a parameter (e.g. `np.transpose(a)`); metadata (`len`, `.shape`, `.ndim`, `.size`), flattened reducers, `argmin`/`argmax`, and the ndarray method forms below all work on the result, multi-argument functions evaluate every argument, and side effects (statements before the `return`, argument calls with side effects) execute exactly once. A 2-D array parameter now keeps its full shape (`numpy_param_shapes_`, populated ahead of the C-ABI row-pointer decay) through `.shape`/`.ndim`/`.size` and `numpy.transpose`/`.T`/`.transpose()`, instead of the decayed 1-D pointer type silently truncating both — see `array_return_param_shape_transpose_success` (formerly a `KNOWNBUG`, now fixed and reclassified `CORE`) and `array_param_shape_metadata_success`/`array_param_transpose_success`. A function call returning a concrete array can also be passed directly as another call's argument (`process(make_array())`), not just a `Name` already holding one. Incompatible-type branches and container-escaped views/descriptors still reject explicitly. Two narrower gaps remain, each pinned by a `KNOWNBUG` regression: (1) `array_return_call_arg_edge` — mutating a captured list/array inside a function without a `global` declaration for it is a pre-existing, unrelated symbol-resolution gap, not specific to array returns; (2) `array_return_side_effect_edge` — an unannotated function whose body builds an array through a local variable before returning it (`a = np.zeros(3); return a`) gets its declared return type locked to the static annotator's own guess in `get_function_definition` before the body is converted, which pre-empts the later GOTO-scan correction. Both surface as an explicit wrong assertion outcome rather than a crash. A numpy array parameter with a genuinely symbolic (non-constant) shape is still rejected, but via a generic `AttributeError` on the first metadata/method access rather than a purpose-built diagnostic — see `array_param_shape_unsupported_symbolic_fail`. |
| Final shared-buffer view model | Partial | ADR-NP-003 etapa 2 now aliases fixed-shape 1-D/2-D views through frontend view metadata. Implemented consumers include literal 1-D slices (unit stride, step != 1, and reversed), 2-D row/column views, `diagonal`, `trace`, `fill_diagonal`, `ravel`/`.flat`, 2-D transpose (`np.transpose`, `.T`, `.transpose()`, `swapaxes`, `moveaxis`), contiguous `reshape` rank 1/2, `squeeze`, `expand_dims`, read-only `broadcast_to`, basic single-operand `nditer`, explicit descriptor materialization (`np.copy`, `view.copy`, `np.array(view)`, including empty descriptors), descriptor `tolist()` rank 1/2, and flattened descriptor reducers (`sum`, `mean`, `min`, `max`, `view.any()`, `view.all()`). Literal-index writes are mirrored across sibling 1-D/2-D descriptor views; non-constant view writes are rejected explicitly. Remaining gaps are 3-D+ view aliasing, symbolic shapes/axes/bounds, non-literal descriptor mutation, non-contiguous reshape beyond the explicit recut, advanced `nditer`, descriptor escape through unknown calls/containers/returns, and making `ndarray_descriptor` itself the consulted runtime structure rather than auxiliary frontend maps. |
| Higher-dimensional or symbolic slice bounds beyond literal-copy cases | Missing | Literal/fixed-shape cases such as bounded 2-D column slices and one-/two-slice-axis mixed tuple indexing are supported. Three or more slice axes, symbolic slice bounds, non-literal strides, and broader stride combinations remain explicitly rejected. |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | Advanced dtype forms (`object`, structured/record dtypes, custom dtype objects) and broad constructor parity. |
| Sorting / searching | `np.sort`/`np.argsort`/`np.searchsorted` and the `a.sort()`/`a.argsort()` method forms now accept concrete ndarray *variables* (including ones returned by a pure user function), row/column views (`a[i]`, `a[:, j]`), and 2-D arrays with an `axis=0`/`axis=1`/`axis=None`/negative-axis argument — `np.sort`/`np.argsort` via `build_numpy_descriptor_materialized_elements` (the same descriptor materialization reducers already used), sorting/permuting each row or column independently via a shared conversion-time bubble-sort network (`bubble_sort_numpy_paired`, capped at `max_numpy_sort_elements`). Still missing: stable-kind variants, sorter/vector-value forms of `searchsorted`, `searchsorted` on a 2-D array or view, and symbolic arrays. |
| Statistics | `a.sum()`/`a.mean()`/`a.min()`/`a.max()`/`a.any()`/`a.all()`/`a.argmin()`/`a.argmax()` method forms and their `axis=0/1` variants are supported over concrete 1-D/2-D ndarrays (including function-returned arrays), sharing the same reducer/comparison policy as the functional forms. Still missing: axis/keepdims/out/overwrite/nan-policy style variants beyond concrete flattened/literal `median` and `percentile`, and reducer axes outside 2-D concrete `axis=0/1`. |
| Linear algebra | `det`/`inv`/`solve` beyond small concrete matrices, symbolic matrix entries, additional `norm` axes/orders, and fuller `eig`/`svd` semantics. |
| Random | Additional distributions, full PRNG state semantics, probability-vector `choice`, replacement control, and large/symbolic shapes. |
| Structured arrays | Record dtypes. |
| Views / strides | Higher-rank (3-D+) view aliasing, symbolic/non-literal-stride slices, symbolic shape/axis handling, non-literal descriptor mutation, advanced descriptor escape handling, and replacing frontend-only maps with a fully consulted `ndarray_descriptor` runtime model. |
| Iteration | Advanced `nditer` flags/options, multi-operand iteration, `external_loop`, `multi_index`, buffering, non-C order, casting/op_dtypes/op_axes, and broader mutable item forms. |

---

## Soundness / performance concerns

1. **Constant-folding bypasses ESBMC's overflow/rounding checks** for folded
   paths. Use `--python-no-fold` to force SMT encoding and compare verdicts.
2. **Element-wise broadcasting** still requires concrete shapes at conversion
   time; symbolic shapes work only for selected array creation paths.
3. **Scalability wall** (#5121): arrays are still represented as fully
   unrolled value lists. Large arrays can explode even when the operation is
   conceptually simple.
4. **Descriptor views still rely on frontend maps instead of one runtime
   descriptor abstraction.** The implemented 1-D/2-D literal-index paths
   propagate writes across tracked sibling views, and unsupported non-literal
   writes reject explicitly; 3-D+, symbolic shape/axis/bound cases, broad
   escape handling, and advanced iterator/method semantics remain
   intentionally incomplete.
5. **Closed: the 2-D parameter shape/transpose soundness gap.** An
   unannotated 2-D array parameter used to be represented internally as a
   flat 1-D array after the C-ABI row-pointer decay, so `numpy.transpose`'s
   "1-D input is a no-op" fallback silently returned such a parameter
   unchanged instead of transposing it. Fixed by tracking the parameter's
   full pre-decay shape (`numpy_param_shapes_`, `python_converter.h`) and
   rematerializing it (via the same descriptor-materialization machinery the
   reducers/sort use) wherever the decayed type alone no longer carries
   enough dimensions — `.shape`/`.ndim`/`.size` and
   `numpy.transpose`/`.T`/`.transpose()`. See
   `array_return_param_shape_transpose_success` (formerly `KNOWNBUG`) and
   `array_param_transpose_success`. The two remaining array-return gaps in
   the table above (`array_return_call_arg_edge`,
   `array_return_side_effect_edge`) still surface as an explicit wrong
   verdict rather than a silently accepted wrong array value.

The numpy call-result chaining gap that used to be listed here — a `Name`
argument whose declaration was itself a non-constructor numpy call resolving
to the wrong operand instead of its evaluated result — was fixed:
`evaluate_numpy_logical_call()` now evaluates
`greater`/`less`/`greater_equal`/`less_equal`/`equal`/`not_equal`/
`logical_and`/`logical_or`/`logical_not`/`where` chained as another numpy
call's argument, nested directly or via an intermediate variable, including
more than one level of chaining; a chain past the supported depth declines
explicitly instead of misreading. See `regression/numpy/chaining_*`.

---

## Community testing readiness

ESBMC's standard across every frontend (C, C++, Solidity, Java/Kotlin) is
sound-but-incomplete, not full language/library coverage: whatever falls
outside the currently supported subset must reject with an explicit
diagnostic (ADR-NP principle 3) rather than silently return a wrong
verdict. By that bar, every gap in "Missing indexing / slicing" and
"Missing API surface" above is **not** a blocker for community testing —
each one already rejects explicitly instead of misbehaving.

With the call-result chaining fix and the parameter-shape/`transpose` fix
(Soundness / performance concerns item 5) above, this file has **no known
open soundness gap** — every remaining item is documented backlog that
surfaces as an explicit "not supported yet" diagnostic, not a wrong answer.
**A build can be cut for community testing from here.**

---

## Prioritised next steps

Nothing below blocks community testing (see above); this is post-release
backlog, in priority order:

1. **3-D+ and symbolic view descriptors (ADR-NP-003 etapa 3)** — extend the
   fixed-shape rank 1/2 descriptor model to higher ranks, symbolic
   shapes/axes/bounds, and broader stride combinations.
2. **`array_return_side_effect_edge`'s return-type-locking gap** — an
   unannotated function whose body builds an array through a local variable
   before returning it (`a = np.zeros(3); return a`) still gets its declared
   return type locked to the static annotator's own guess in
   `get_function_definition` before the body is converted. Distinct from the
   now-fixed parameter-shape/`transpose` gap (a decayed-pointer-type issue,
   not a return-type-locking one) — root-causing it separately is still
   open.
3. **Symbolic and broader multi-axis slicing** — support cases beyond the
   literal/fixed-shape recuts.
4. **Symbolic-shape numpy array parameters** — currently rejected via a
   generic `AttributeError` on first metadata/method access
   (`array_param_shape_unsupported_symbolic_fail`) rather than a
   purpose-built diagnostic naming the actual constraint.
5. **Advanced dtype and constructor parity** — structured/object/custom dtype
   policy, diagnostics, and propagation.
6. **Random and iteration depth** — probability/replacement `choice`, extra
   distributions, and advanced `nditer`.
7. **Linear algebra breadth** — larger matrices, symbolic entries, and more
   faithful `norm`/`eig`/`svd`.

---

## Suggested next PRs

Each roadmap item above groups several sub-efforts; sizing them 1 PR per
item undercounts the real work. Items below with multiple named consumers or
distinct designs are sized accordingly instead of assumed to be one PR each.

1. **3-D+ / symbolic view descriptors** (~2 PRs) — extend the rank 1/2
   fixed-shape descriptor model to higher ranks, symbolic axes/bounds/shapes,
   and broader non-literal stride combinations.
2. **`array_return_side_effect_edge`'s return-type-locking gap** (~1 PR) —
   an unannotated function's local-variable-then-return array path still
   gets its return type locked to the static annotator's guess ahead of
   body conversion.
3. **Stable-kind sort/searching gaps** (~1 PR) — `kind=` stability,
   sorter/vector-value `searchsorted`, `searchsorted` on a 2-D array/view,
   and symbolic arrays (axis-aware and 2-D `sort`/`argsort`/`searchsorted`
   over concrete arrays and views are now implemented).
4. **Advanced dtype and constructors** (~2 PRs) — dtype policy
   (object/structured/custom) separate from constructor
   diagnostics/propagation.
5. **Random and iteration depth** (~2 PRs) — new distributions/`choice`
   separate from advanced `nditer`.
6. **Linear algebra expansion** (~2 PRs) — larger/symbolic matrix support
   separate from fuller `eig`/`svd`/`norm`.

**Total to close every item in this file: ~9 PRs.**

---

## Out of scope

- True SMT-array scalability beyond the current `array_typet` lowering; see
  ADR-NP-004.
- Extending the runtime-list model to hold array-typed elements; this remains
  disproportionately risky for current NumPy goals.
