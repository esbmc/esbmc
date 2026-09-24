# ESBMC NumPy — Remaining Work

**Updated:** 2026-09-24.

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
| General NumPy array returns from user functions | Partial | A user function can now return a concrete array/view/descriptor built from `np.array`/`np.zeros`/`np.ones`/`np.full`, a bare parameter, a subscript/subarray of a parameter, or a supported descriptor call over a parameter (e.g. `np.transpose(a)`); metadata (`len`, `.shape`, `.ndim`, `.size`), flattened reducers, `argmin`/`argmax`, and the ndarray method forms below all work on the result, multi-argument functions evaluate every argument, and side effects (statements before the `return`, argument calls with side effects) execute exactly once. A 2-D array parameter now keeps its full shape (`numpy_param_shapes_`, populated ahead of the C-ABI row-pointer decay) through `.shape`/`.ndim`/`.size` and `numpy.transpose`/`.T`/`.transpose()`, instead of the decayed 1-D pointer type silently truncating both. **Closed this cycle:** a function that builds its returned array through a local variable before `return`ing it (`a = np.zeros(3); a[0] = 5; return a`) used to get its declared return type locked to the static annotator's own `list[float]` guess for the numpy constructor call ahead of body conversion — the annotator now declines that guess specifically for a name the function returns directly, letting the existing GOTO-scan fallback type it correctly from the converted body instead; branches that build the returned array from constructor calls of different shapes reject explicitly rather than crashing the branch-merge. See `array_return_local_construct_success`, `array_return_local_branch_success`, `array_return_local_branch_incompatible_fail`, and the promoted `array_return_side_effect_edge` (formerly `KNOWNBUG`, now `CORE`). One narrower gap remains, pinned by a `KNOWNBUG` regression: `array_return_call_arg_edge` — mutating a captured list/array inside a function without a `global` declaration for it is a pre-existing, unrelated symbol-resolution gap, not specific to array returns. A numpy array parameter with a genuinely symbolic (non-constant) shape is now rejected with a purpose-built diagnostic (`TypeError: numpy array parameter shape must be concrete for .shape/.ndim/.size/len()/transpose()/sort()/argsort()`) when the function body actually reads one of those — see `array_param_shape_unsupported_symbolic_fail`, `array_param_shape_symbolic_metadata_fail`, `array_param_shape_symbolic_transpose_fail`, `array_param_shape_symbolic_sort_fail`; `len()` on such a parameter already resolves soundly through a different path and is deliberately left alone (`array_param_shape_symbolic_len_success`). |
| Final shared-buffer view model | Partial | ADR-NP-003 etapa 2 now aliases fixed-shape 1-D/2-D views through frontend view metadata. Implemented consumers include literal 1-D slices (unit stride, step != 1, and reversed), 2-D row/column views, `diagonal`, `trace`, `fill_diagonal`, `ravel`/`.flat`, 2-D transpose (`np.transpose`, `.T`, `.transpose()`, `swapaxes`, `moveaxis`), contiguous `reshape` rank 1/2, `squeeze`, `expand_dims`, read-only `broadcast_to`, basic single-operand `nditer`, explicit descriptor materialization (`np.copy`, `view.copy`, `np.array(view)`, including empty descriptors), descriptor `tolist()` rank 1/2, and flattened descriptor reducers (`sum`, `mean`, `min`, `max`, `view.any()`, `view.all()`). Literal-index writes are mirrored across sibling 1-D/2-D descriptor views; non-constant view writes are rejected explicitly. Remaining gaps are 3-D+ view aliasing, symbolic shapes/axes/bounds, non-literal descriptor mutation, non-contiguous reshape beyond the explicit recut, advanced `nditer`, descriptor escape through unknown calls/containers/returns, and making `ndarray_descriptor` itself the consulted runtime structure rather than auxiliary frontend maps. |
| Higher-dimensional or symbolic slice bounds beyond literal-copy cases | Missing | Literal/fixed-shape cases such as bounded 2-D column slices and one-/two-slice-axis mixed tuple indexing are supported. Three or more slice axes, symbolic slice bounds, non-literal strides, and broader stride combinations remain explicitly rejected. |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | Advanced dtype forms (`object`, structured/record dtypes, custom dtype objects) still reject explicitly; broad constructor parity beyond `zeros`/`ones`/`full`/`array`/`eye`/`identity`/`linspace`/`arange`. **Closed this cycle:** a `dtype=` keyword (a builtin `int`/`float`/`bool` or a numpy alias like `np.int32`) on `zeros`/`ones`/`full`/`eye`/`identity`/`linspace`/`arange` used to make every consumer that re-derives the constructor from its call node (`transpose`, `flatten`, `ravel`, `sort`, `argsort`, `searchsorted`, and an inline-call reducer argument) decline outright with a generic "unsupported keywords" diagnostic, even though the same `dtype=` already worked on a plain `x = np.zeros(..., dtype=...)` assignment; the shared re-derivation (`materialize_numpy_constructor_array`) now accepts a lone `dtype=` keyword and normalizes/casts elements through the same validated path the assignment case uses, still rejecting `object`/complex/non-literal dtype and any other keyword explicitly — see `transpose_constructor_dtype_*`, `flatten_constructor_dtype_zeros_success`, `ravel_constructor_dtype_eye_edge`, `sort_constructor_dtype_full_success`, `argsort_constructor_dtype_full_success`, `searchsorted_constructor_dtype_full_success`, `sum_constructor_dtype_zeros_success`, `mean_constructor_dtype_eye_success`, and the `constructor_dtype_object_fail`/`complex_fail`/`nonliteral_fail` protective pins. `np.zeros_like`/`np.ones_like`/`np.full_like` now accept a `dtype=` override the same way real numpy does (the result casts to the given dtype instead of inheriting the base array's) — see `zeros_like_dtype_override_success`, `ones_like_dtype_override_bool_edge`, `full_like_dtype_override_success`, `like_creation_dtype_object_fail`. **Newly discovered, out of scope for this cycle:** chaining a transpose/flatten method directly onto a constructor call with no intermediate variable (`np.eye(3).transpose()`, `np.zeros((2,2)).flatten()`, `np.zeros((2,3)).T`) produces a wrong `NONDET` result instead of a diagnostic — pre-existing and unrelated to `dtype=` (reproduces identically without it); binding the constructor to a variable first works. `.size` on an `eye`/`full`/`identity`/`linspace`-backed array (`np.zeros((2,3)).size` works, but the `full`/`eye` family does not) crashes the backend (`ERROR: Unexpected type in int/ptr typecast`) instead of rejecting cleanly — also pre-existing and unrelated to `dtype=`. Neither is pinned by a regression yet. |
| Sorting / searching | `np.sort`/`np.argsort`/`np.searchsorted` and the `a.sort()`/`a.argsort()` method forms accept concrete ndarray *variables* (including ones returned by a pure user function, direct or via a local variable), row/column views (`a[i]`, `a[:, j]` — both axes, including through `np.searchsorted`), and 2-D arrays with an `axis` argument, positional or `axis=` keyword (never both). `numpy.searchsorted()` accepts a vector of values (`np.searchsorted(a, [2, 6])`, a literal list/tuple or a `Name` bound to one) in addition to a scalar, returning an index per value; and a `sorter=` argument (positional or keyword, a literal index array or `np.argsort(a)`/`a.argsort()` computed directly) over an AST-literal array, validated as a genuine permutation of the input's own index range and letting an otherwise-unsorted array be searched. `kind='stable'`/`kind='mergesort'`/`kind=None` (and, for the ndarray method forms, `stable=True`) are accepted no-ops on `np.sort`/`np.argsort`/`a.sort()`/`a.argsort()`, since the shared conversion-time bubble sort is already stable — any other `kind` still rejects explicitly with a diagnostic naming the supported values. **Closed this cycle:** `numpy.searchsorted()`'s array argument now also resolves through the same descriptor-materialization path `sort`/`argsort` already use for a `Name` — a variable bound to a function that returns its array through a local variable (`def make(): a = np.zeros(3); return a`), or a direct call to one (`np.searchsorted(make(), ...)`, evaluated exactly once even under a discarded LHS-type-inference probe pass), now resolves and computes the position via an exprt-level comparison count (the same style `sort`/`argsort`'s bubble sort uses for its own comparisons), rather than trying to read a compile-time literal back out of it — see `array_return_local_then_searchsorted_direct_success`, `_vector_success`, `_wrong_result_fail`, and the promoted `array_return_local_then_searchsorted_edge`. Still missing: `searchsorted` on a genuine 2-D array (as opposed to a 1-D row/column view of one), symbolic arrays, and — **newly discovered, out of scope for this cycle** — `sorter=` combined with a descriptor-resolved array (the sorter mechanism still needs a literal `arr_arg` to validate/apply against; an exprt-level stable-sort permutation would be needed to lift that) — see `array_return_local_then_searchsorted_sorter_edge` (`KNOWNBUG`). |
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
5. **Closed: the local-array-return return-type-locking soundness gap.** A
   function returning a numpy array it built through a local variable
   (`a = np.zeros(3); return a`) had its declared return type locked to a
   generic `PyListObject*` ahead of body conversion, because the static
   annotator resolved the local variable's own `np.zeros(...)` binding
   through the numpy operational model's declared `list[float]` signature —
   a mismatch invisible to the model, since the model has no notion of the
   converter's later concrete array type. With a symbolic branch condition
   this let a real out-of-bounds access go undetected (`VERIFICATION
   SUCCESSFUL`) instead of being rejected — see
   `array_return_divergent_branch_shape_fail` (from a related, already-fixed
   nested-return-inference gap) for the same class of unsoundness. Fixed by
   having the annotator decline that generic guess specifically when the
   name being assigned is the one the enclosing function returns directly,
   so the existing GOTO-scan fallback types the function from its actual
   converted body instead. See `array_return_local_construct_success`,
   `array_return_local_branch_success`, and the promoted
   `array_return_side_effect_edge`.
6. **Closed: the 2-D parameter shape/transpose soundness gap.** An
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
   `array_param_transpose_success`. The remaining array-return gap in the
   table above (`array_return_call_arg_edge`) still surfaces as an explicit
   wrong verdict rather than a silently accepted wrong array value.

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

With the call-result chaining fix, the parameter-shape/`transpose` fix, and
the local-array-return return-type-locking fix (Soundness / performance
concerns items 5–6) above, this file has **no known open soundness gap** —
every remaining item is documented backlog that surfaces as an explicit "not
supported yet" diagnostic, not a wrong answer. **A build can be cut for
community testing from here.**

---

## Prioritised next steps

Nothing below blocks community testing (see above); this is post-release
backlog, in priority order:

1. **3-D+ and symbolic view descriptors (ADR-NP-003 etapa 3)** — extend the
   fixed-shape rank 1/2 descriptor model to higher ranks, symbolic
   shapes/axes/bounds, and broader stride combinations.
2. **Symbolic and broader multi-axis slicing** — support cases beyond the
   literal/fixed-shape recuts.
3. **`numpy.searchsorted()`'s remaining gaps** — a genuine 2-D array input
   (as opposed to a row/column view), symbolic arrays, and `sorter=` over a
   descriptor-resolved array (currently
   `array_return_local_then_searchsorted_sorter_edge`, a `KNOWNBUG`).
4. **Chained method call on a bare constructor without a variable**
   (`np.eye(3).transpose()`, `np.zeros((2,2)).flatten()`,
   `np.zeros((2,3)).T`) — produces a wrong `NONDET` result instead of a
   diagnostic; not yet pinned by a regression. Unrelated to `dtype=`.
5. **`.size` on an `eye`/`full`/`identity`/`linspace`-backed array** crashes
   the backend instead of rejecting cleanly; not yet pinned by a regression.
   Unrelated to `dtype=`.
6. **Advanced dtype and constructor parity** — structured/object/custom dtype
   policy, diagnostics, and propagation (constructor `dtype=` for the
   already-supported builtin/numpy-alias dtypes is closed — see above).
7. **Random and iteration depth** — probability/replacement `choice`, extra
   distributions, and advanced `nditer`.
8. **Linear algebra breadth** — larger matrices, symbolic entries, and more
   faithful `norm`/`eig`/`svd`.

---

## Suggested next PRs

Each roadmap item above groups several sub-efforts; sizing them 1 PR per
item undercounts the real work. Items below with multiple named consumers or
distinct designs are sized accordingly instead of assumed to be one PR each.

1. **3-D+ / symbolic view descriptors** (~2 PRs) — extend the rank 1/2
   fixed-shape descriptor model to higher ranks, symbolic axes/bounds/shapes,
   and broader non-literal stride combinations.
2. **`numpy.searchsorted()`'s remaining gaps** (~1 PR) — a genuine 2-D array
   input, symbolic arrays, and `sorter=` over a descriptor-resolved array
   (scalar/vector local-array-return resolution is now implemented).
3. **Chained-constructor-call and `.size` fixes** (~1 PR) — the
   `np.eye(3).transpose()`-without-a-variable gap and the `eye`/`full`-family
   `.size` crash, both newly discovered this cycle and both pre-existing
   (unrelated to `dtype=`).
4. **Advanced dtype and constructors** (~2 PRs) — dtype policy
   (object/structured/custom) separate from constructor
   diagnostics/propagation (builtin/numpy-alias `dtype=` materialization is
   now closed).
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
