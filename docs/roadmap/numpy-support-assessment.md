# ESBMC NumPy — Remaining Work

**Updated:** 2026-10-01.

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
| Final shared-buffer view model | Partial | ADR-NP-003 etapa 2 now aliases fixed-shape 1-D/2-D views through frontend view metadata, and the current branch extends selected fixed-shape rank 3/4 recuts. Implemented consumers include literal 1-D slices (unit stride, step != 1, and reversed), 2-D row/column views, rank-3 subarray row views with aliasing, chained rank-3 subarray writes, rank-3 descriptor `tolist()`, `diagonal`, `trace`, `fill_diagonal`, `ravel`/`.flat`, 2-D transpose (`np.transpose`, `.T`, `.transpose()`), fixed-shape rank-3 `transpose`/`swapaxes`/`moveaxis` materialization, contiguous `reshape` rank 1/2/3, `squeeze`, `expand_dims`, read-only `broadcast_to`, basic single-operand `nditer`, explicit descriptor materialization (`np.copy`, `view.copy`, `np.array(view)`, including empty descriptors), descriptor `tolist()` rank 1/2/3, and flattened descriptor reducers (`sum`, `mean`, `min`, `max`, `view.any()`, `view.all()`). Literal-index writes are mirrored across sibling 1-D/2-D descriptor views; rank-3 subarray writes alias through pointer views where supported. Remaining gaps are full 3-D+ descriptor aliasing for every view producer, symbolic shapes/axes, non-literal descriptor mutation, non-contiguous reshape beyond the explicit recut, advanced `nditer`, descriptor escape through unknown calls/containers/returns, and making `ndarray_descriptor` itself the consulted runtime structure rather than auxiliary frontend maps. |
| Higher-dimensional or symbolic slice bounds beyond literal-copy cases | Partial | Literal/fixed-shape cases such as bounded 2-D column slices, one-/two-slice-axis mixed tuple indexing, and selected 3-D mixed tuple slice shapes are supported. Symbolic positive and negative 1-D slice bounds are normalized for non-empty slices. Remaining gaps: symbolic empty slices still require a separate logical length from the physical VLA allocation, symbolic/non-literal strides are rejected, three or more slice axes remain outside the current recut, and broader stride combinations are incomplete. |

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
| Views / strides | Full higher-rank descriptor aliasing beyond the selected rank-3 subarray paths, symbolic empty slice logical lengths, symbolic/non-literal-stride slices, symbolic shape/axis handling, non-literal descriptor mutation, advanced descriptor escape handling, and replacing frontend-only maps with a fully consulted `ndarray_descriptor` runtime model. |
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
4. **Descriptor views still rely on frontend maps instead of one runtime
   descriptor abstraction.** The implemented 1-D/2-D literal-index paths
   propagate writes across tracked sibling views, and selected rank-3 subarray
   paths alias through pointer views; unsupported non-literal writes reject
   explicitly. Full 3-D+ descriptor aliasing, symbolic shapes/axes, symbolic
   empty-slice logical lengths, broad escape handling, and advanced
   iterator/method semantics remain intentionally incomplete.
---

## Community testing readiness

ESBMC's standard across every frontend (C, C++, Solidity, Java/Kotlin) is
sound-but-incomplete, not full language/library coverage: whatever falls
outside the currently supported subset must reject with an explicit
diagnostic (ADR-NP principle 3) rather than silently return a wrong
verdict. Gaps in "Missing indexing / slicing" and "Missing API
surface" should reject explicitly; any known false alarm must be listed as a
separate gap instead of treated as supported behavior.

This file has **no known open NumPy unsound-success gap**. Remaining NumPy
items are documented backlog: unsupported cases should reject explicitly,
and known false alarms are listed as gaps instead of treated as supported
behavior. **A build can be cut for community testing from here.**

`regression/numpy` pins eight `KNOWNBUG` tests (2026-10-01):

| test | expected | today |
|---|---|---|
| `e`, `round2`, `view_branch_registration_conflict_knownbug` | SUCCESSFUL | `VERIFICATION FAILED` (false alarm) |
| `isclose`, `nextafter`, `remainder` | SUCCESSFUL | internal error (an expression dump, no verdict) |
| `array_return_call_arg_edge` | SUCCESSFUL | rejected: `'int' object is not subscriptable` |
| `det2` | FAILED | rejected: `numpy.linalg.det supports only 2x2 and 3x3 matrices` |

---

## Prioritised next steps

Nothing below blocks community testing (see above); this is post-release
backlog, in priority order:

1. **Finish descriptor-backed 3-D+ views (ADR-NP-003 etapa 3)** — replace the
   selected rank-3/4 recuts with one runtime-consulted descriptor path for all
   supported producers and consumers.
2. **Symbolic and broader multi-axis slicing** — add symbolic empty-slice
   logical lengths, symbolic/non-literal strides, and broader 3-D+ multi-axis
   slicing beyond the currently supported literal recuts.
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

1. **Complete 3-D+ / symbolic view descriptors** (~2 PRs) — finish the
   descriptor-backed runtime path for all view producers/consumers, symbolic
   axes/shapes, symbolic empty-slice lengths, and non-literal stride
   combinations.
2. **Advanced dtype and constructors** (~2 PRs) — dtype policy
   (object/structured/custom) separate from constructor
   diagnostics/propagation.
3. **Random and iteration depth** (~2 PRs) — new distributions/`choice`
   separate from advanced `nditer`.
4. **Linear algebra expansion** (~2 PRs) — larger/symbolic matrix support
   separate from fuller `eig`/`svd`/`norm`.

**Total to close every item in this file: ~6 PRs.**

---

## Out of scope

- True SMT-array scalability beyond the current `array_typet` lowering; see
  ADR-NP-004.
- Extending the runtime-list model to hold array-typed elements; this remains
  disproportionately risky for current NumPy goals.
