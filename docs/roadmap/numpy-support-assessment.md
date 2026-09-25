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
| General NumPy array returns from user functions | Partial | Concrete array/view/descriptor returns from supported constructors, bare parameters, parameter subscript/subarrays, and supported descriptor calls over parameters are implemented. Remaining gap: `array_return_call_arg_edge` is still pinned as `KNOWNBUG`; mutating a captured list/array inside a function without a `global` declaration is a pre-existing symbol-resolution gap, not specific to array returns. Genuinely symbolic parameter shapes are rejected when unsupported metadata/view consumers need concrete shape information; `len()` on such parameters remains supported through its independent path. |
| Final shared-buffer view model | Partial | ADR-NP-003 etapa 2 now aliases fixed-shape 1-D/2-D views through frontend view metadata. Implemented consumers include literal 1-D slices (unit stride, step != 1, and reversed), 2-D row/column views, `diagonal`, `trace`, `fill_diagonal`, `ravel`/`.flat`, 2-D transpose (`np.transpose`, `.T`, `.transpose()`, `swapaxes`, `moveaxis`), contiguous `reshape` rank 1/2, `squeeze`, `expand_dims`, read-only `broadcast_to`, basic single-operand `nditer`, explicit descriptor materialization (`np.copy`, `view.copy`, `np.array(view)`, including empty descriptors), descriptor `tolist()` rank 1/2, and flattened descriptor reducers (`sum`, `mean`, `min`, `max`, `view.any()`, `view.all()`). Literal-index writes are mirrored across sibling 1-D/2-D descriptor views; non-constant view writes are rejected explicitly. Remaining gaps are 3-D+ view aliasing, symbolic shapes/axes/bounds, non-literal descriptor mutation, non-contiguous reshape beyond the explicit recut, advanced `nditer`, descriptor escape through unknown calls/containers/returns, and making `ndarray_descriptor` itself the consulted runtime structure rather than auxiliary frontend maps. |
| Higher-dimensional or symbolic slice bounds beyond literal-copy cases | Missing | Literal/fixed-shape cases such as bounded 2-D column slices and one-/two-slice-axis mixed tuple indexing are supported. Three or more slice axes, symbolic slice bounds, non-literal strides, and broader stride combinations remain explicitly rejected. |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | Advanced dtype forms (`object`, structured/record dtypes, custom dtype objects) still reject explicitly; broad constructor parity beyond `zeros`/`ones`/`full`/`array`/`eye`/`identity`/`linspace`/`arange`. `.size` on `np.eye`, `np.identity`, and `np.full` results currently false-alarms on valid assertions about the array size; `np.linspace(...).size` works. This gap is not yet pinned by a regression. |
| Sorting / searching | `searchsorted` on a genuine 2-D array (as opposed to a 1-D row/column view of one), symbolic arrays, and a literal (non-`argsort`) `sorter=` index array over a descriptor-resolved array. |
| Direct constructor chaining | Several methods work directly on supported constructor calls (`sum`, `mean`, `min`, `max`, `argsort`, `searchsorted`, `flatten`, `ravel`, `transpose`, `.T`, and selected `reshape` cases), but `argmin`, `argmax`, `diagonal`, `prod`, `std`, `var`, and `np.array(...).reshape(...)` still reject with the explicit "assign the constructor's result to a variable first" diagnostic. |
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
---

## Community testing readiness

ESBMC's standard across every frontend (C, C++, Solidity, Java/Kotlin) is
sound-but-incomplete, not full language/library coverage: whatever falls
outside the currently supported subset must reject with an explicit
diagnostic (ADR-NP principle 3) rather than silently return a wrong
verdict. Most gaps in "Missing indexing / slicing" and "Missing API
surface" already reject explicitly; the known exception is the constructor
`.size` false-alarm gap listed above.

This file has **no known open NumPy unsound-success gap**. Remaining NumPy
items are documented backlog: unsupported cases should reject explicitly,
and known false alarms are listed as gaps instead of treated as supported
behavior. **A build can be cut for community testing from here.**

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
   (as opposed to a row/column view), symbolic arrays, and a literal
   (non-`argsort`) `sorter=` index array over a descriptor-resolved array.
4. **`.size` on `eye`/`full`/`identity` results** false-alarms on valid
   size assertions; not yet pinned by a regression. Unrelated to `dtype=`.
5. **Direct constructor chaining parity** — close the remaining explicit
   rejections for `argmin`, `argmax`, `diagonal`, `prod`, `std`, `var`, and
   `np.array(...).reshape(...)`.
6. **Advanced dtype and constructor parity** — structured/object/custom dtype
   policy, diagnostics, and propagation.
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
   input, symbolic arrays, and a literal (non-`argsort`) `sorter=` index
   array over a descriptor-resolved array.
3. **`.size` fix** (~1 PR) — the `eye`/`full`/`identity` false-alarm gap,
   pre-existing and unrelated to `dtype=`.
4. **Direct constructor chaining parity** (~1 PR) — close the remaining
   explicit method-chain rejections listed above.
5. **Advanced dtype and constructors** (~2 PRs) — dtype policy
   (object/structured/custom) separate from constructor
   diagnostics/propagation.
6. **Random and iteration depth** (~2 PRs) — new distributions/`choice`
   separate from advanced `nditer`.
7. **Linear algebra expansion** (~2 PRs) — larger/symbolic matrix support
   separate from fuller `eig`/`svd`/`norm`.

**Total to close every item in this file: ~10 PRs.**

---

## Out of scope

- True SMT-array scalability beyond the current `array_typet` lowering; see
  ADR-NP-004.
- Extending the runtime-list model to hold array-typed elements; this remains
  disproportionately risky for current NumPy goals.
