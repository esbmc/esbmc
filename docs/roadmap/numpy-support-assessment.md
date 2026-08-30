# ESBMC NumPy — Remaining Work

**Updated:** 2026-08-29.

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
| General NumPy array returns from user functions | Partial | A user function can now return a concrete array/view/descriptor built from `np.array`/`np.zeros`/`np.ones`/`np.full`, a bare parameter, a subscript/subarray of a parameter, or a supported descriptor call over a parameter (e.g. `np.transpose(a)`); metadata (`len`, `.shape`, `.ndim`, `.size`), flattened reducers, `argmin`/`argmax`, and the ndarray method forms below all work on the result, multi-argument functions evaluate every argument, and side effects (statements before the `return`, argument calls with side effects) execute exactly once. Incompatible-type branches and container-escaped views/descriptors still reject explicitly. Three narrower gaps remain, each pinned by a `KNOWNBUG` regression: (1) `array_return_call_arg_edge` — mutating a captured list/array inside a function without a `global` declaration for it is a pre-existing, unrelated symbol-resolution gap, not specific to array returns; (2) `array_return_side_effect_edge` — an unannotated function whose body builds an array through a local variable before returning it (`a = np.zeros(3); return a`) gets its declared return type locked to the static annotator's own guess in `get_function_definition` before the body is converted, which pre-empts the later GOTO-scan correction; (3) `array_return_param_shape_transpose_knownbug` — an unannotated 2-D array parameter is typed as a flat 1-D array within the callee's own body, so `numpy.transpose`'s "1-D input is a no-op" fallback silently returns it unchanged whenever the caller-side return-value inlining that normally masks this can't apply (e.g. a call the inliner declines, or a second, function-local alias for `numpy`). (1) and (2) surface as an explicit wrong assertion outcome rather than a crash; (3) is the one case here that can produce a **silently wrong array value** instead of an explicit rejection — see the soundness section below. |
| Final shared-buffer view model | Partial | ADR-NP-003 etapa 2 now aliases fixed-shape 1-D/2-D views through frontend view metadata. Implemented consumers include literal 1-D slices (unit stride, step != 1, and reversed), 2-D row/column views, `diagonal`, `trace`, `fill_diagonal`, `ravel`/`.flat`, 2-D transpose (`np.transpose`, `.T`, `.transpose()`, `swapaxes`, `moveaxis`), contiguous `reshape` rank 1/2, `squeeze`, `expand_dims`, read-only `broadcast_to`, basic single-operand `nditer`, explicit descriptor materialization (`np.copy`, `view.copy`, `np.array(view)`, including empty descriptors), descriptor `tolist()` rank 1/2, and flattened descriptor reducers (`sum`, `mean`, `min`, `max`, `view.any()`, `view.all()`). Literal-index writes are mirrored across sibling 1-D/2-D descriptor views; non-constant view writes are rejected explicitly. Remaining gaps are 3-D+ view aliasing, symbolic shapes/axes/bounds, non-literal descriptor mutation, non-contiguous reshape beyond the explicit recut, advanced `nditer`, descriptor escape through unknown calls/containers/returns, and making `ndarray_descriptor` itself the consulted runtime structure rather than auxiliary frontend maps. |
| Higher-dimensional or symbolic slice bounds beyond literal-copy cases | Missing | Literal/fixed-shape cases such as bounded 2-D column slices and one-/two-slice-axis mixed tuple indexing are supported. Three or more slice axes, symbolic slice bounds, non-literal strides, and broader stride combinations remain explicitly rejected. |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | Advanced dtype forms (`object`, structured/record dtypes, custom dtype objects) and broad constructor parity. |
| Sorting / searching | `np.sort`/`np.argsort`/`np.searchsorted`/`a.searchsorted()` now accept concrete ndarray *variables* (including ones returned by a pure user function), not just inline literals, and `a.sort()`/`a.argsort()` are supported as in-place/copy method forms over 1-D concrete arrays. Still missing: axis-aware and stable-kind variants, `sort`/`argsort`/`searchsorted` on 2-D arrays, sorter/vector-value forms of `searchsorted`, and symbolic arrays. |
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
5. **One open soundness gap in general array returns.** An unannotated
   function parameter typed as a 2-D array is represented internally as a
   flat 1-D array (a pre-existing gap in this frontend's parameter shape
   propagation, unrelated to array returns specifically). `numpy.transpose`'s
   existing "1-D input is a no-op" fallback then silently returns such a
   parameter unchanged instead of transposing it, whenever the caller-side
   return-value inlining that normally masks this (by substituting the
   caller's own, correctly-shaped argument before the callee's body ever
   runs) does not apply — e.g. a second, function-local alias for `numpy`,
   or any other call shape the inliner declines. Pinned as `KNOWNBUG` in
   `regression/numpy/array_return_param_shape_transpose_knownbug`. The two
   other array-return gaps in the table above (`array_return_call_arg_edge`,
   `array_return_side_effect_edge`) surface as an explicit wrong verdict
   rather than a silently accepted wrong array value.

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

With the call-result chaining fix above, the only known soundness gap left in
this file is the narrow parameter-shape/`transpose` case in "Soundness /
performance concerns" item 5, pinned as a `KNOWNBUG` regression rather than
left silently passing. **A build can be cut for community testing from
here** — everything else remaining is documented backlog that surfaces as an
explicit "not supported yet" diagnostic, not a wrong answer; testers hitting
the one open gap will see a plausible-looking but incorrect transposed array
rather than a crash or rejection, so it is worth flagging in release notes
for anyone exercising array-returning functions with unannotated 2-D
parameters.

---

## Prioritised next steps

Nothing below blocks community testing (see above); this is post-release
backlog, in priority order:

1. **3-D+ and symbolic view descriptors (ADR-NP-003 etapa 3)** — extend the
   fixed-shape rank 1/2 descriptor model to higher ranks, symbolic
   shapes/axes/bounds, and broader stride combinations.
2. **Parameter shape propagation for unannotated 2-D array parameters** —
   root-cause why such a parameter is typed as flat 1-D within the callee's
   own body (soundness item 5); this also underlies
   `array_return_side_effect_edge`'s return-type-locking gap, since both
   trace back to the static annotator's guess winning over the function
   body's real shape.
3. **Symbolic and broader multi-axis slicing** — support cases beyond the
   literal/fixed-shape recuts.
4. **Axis-aware and 2-D `sort`/`argsort`/`searchsorted`** — extend the
   now-concrete-variable 1-D recut to `axis=`, stable-kind kwargs, and 2-D
   arrays.
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
2. **Parameter shape propagation** (~1 PR) — fix the unannotated-2-D-array-
   parameter typing gap (soundness item 5); closes the one remaining silent
   wrong-answer case plus `array_return_side_effect_edge`.
3. **Axis-aware and 2-D sort/searching** (~1 PR) — `axis=`, stable-kind
   kwargs, and 2-D `sort`/`argsort`/`searchsorted`.
4. **Advanced dtype and constructors** (~2 PRs) — dtype policy
   (object/structured/custom) separate from constructor
   diagnostics/propagation.
5. **Random and iteration depth** (~2 PRs) — new distributions/`choice`
   separate from advanced `nditer`.
6. **Linear algebra expansion** (~2 PRs) — larger/symbolic matrix support
   separate from fuller `eig`/`svd`/`norm`.

**Total to close every item in this file: ~10 PRs.**

---

## Out of scope

- True SMT-array scalability beyond the current `array_typet` lowering; see
  ADR-NP-004.
- Extending the runtime-list model to hold array-typed elements; this remains
  disproportionately risky for current NumPy goals.
