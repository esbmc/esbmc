# ESBMC NumPy — Remaining Work

**Updated:** 2026-08-25.

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
| General NumPy array returns from user functions | Missing | Only the narrow identity-return pattern is supported. Non-trivial returns such as `def f(a): return a[0]`, functions with multiple parameters, and functions with more than one statement still need a real fix in the assignment/type-inference pipeline. Previous attempts hit double conversion in `create_symbol_for_unannotated_assign` / `get_var_assign` and wrapper-type confusion before variable type selection. |
| Final shared-buffer view model | Partial | ADR-NP-003 etapa 2 now aliases fixed-shape 1-D/2-D views through frontend view metadata. Implemented consumers include literal 1-D slices (unit stride, step != 1, and reversed), 2-D row/column views, `diagonal`, `trace`, `fill_diagonal`, `ravel`/`.flat`, 2-D transpose (`np.transpose`, `.T`, `.transpose()`, `swapaxes`, `moveaxis`), contiguous `reshape` rank 1/2, `squeeze`, `expand_dims`, read-only `broadcast_to`, basic single-operand `nditer`, explicit descriptor materialization (`np.copy`, `view.copy`, `np.array(view)`), descriptor `tolist()` rank 1/2, and flattened descriptor reducers (`sum`, `mean`, `min`, `max`, `view.any()`, `view.all()`). Remaining gaps are 3-D+ view aliasing, symbolic shapes/axes/bounds, non-contiguous reshape beyond the explicit recut, advanced `nditer`, descriptor escape through unknown calls/containers/returns, and making `ndarray_descriptor` itself the consulted runtime structure rather than auxiliary frontend maps. |
| Higher-dimensional or symbolic slice bounds beyond literal-copy cases | Missing | Literal/fixed-shape cases such as bounded 2-D column slices and one-/two-slice-axis mixed tuple indexing are supported. Three or more slice axes, symbolic slice bounds, non-literal strides, and broader stride combinations remain explicitly rejected. |
| Remaining array method forms | Partial | `view.tolist()` is supported for descriptor views rank 1/2, and `view.any()`/`view.all()` are supported for descriptor views rank 1/2. Plain ndarray `a.tolist()`, plain ndarray `a.any()`/`a.all()`, `a.sort()`, and `a.argsort()` remain separate method-form gaps. `a.sort()` is in-place while `np.sort(a)` returns a copy; `a.argsort()` needs `np.argsort()` to support variables first. |

---

## Missing API surface

| Category | Missing items |
|---|---|
| Array creation | Advanced dtype forms (`object`, structured/record dtypes, custom dtype objects) and broad constructor parity. |
| Sorting / searching | Axis-aware and stable-kind variants, `sort`/`argsort` kwargs beyond the supported concrete 1-D recut, and symbolic arrays. |
| Statistics | Axis/keepdims/out/overwrite/nan-policy style variants beyond concrete flattened/literal `median` and `percentile`. |
| Linear algebra | `det`/`inv`/`solve` beyond small concrete matrices, symbolic matrix entries, additional `norm` axes/orders, and fuller `eig`/`svd` semantics. |
| Random | Additional distributions, full PRNG state semantics, probability-vector `choice`, replacement control, and large/symbolic shapes. |
| Structured arrays | Record dtypes. |
| Views / strides | Higher-rank (3-D+) view aliasing, symbolic/non-literal-stride slices, symbolic shape/axis handling, advanced descriptor escape handling, and replacing frontend-only maps with a fully consulted `ndarray_descriptor` runtime model. |
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
   descriptor abstraction.** The implemented 1-D/2-D cases alias correctly,
   but 3-D+, symbolic shape/axis/bound cases, broad escape handling, and
   advanced iterator/method semantics remain intentionally incomplete.

No known soundness bugs remain open (the numpy call-result chaining gap that
used to be listed here — a `Name` argument whose declaration was itself a
non-constructor numpy call resolving to the wrong operand instead of its
evaluated result — was fixed: `evaluate_numpy_logical_call()` now evaluates
`greater`/`less`/`greater_equal`/`less_equal`/`equal`/`not_equal`/
`logical_and`/`logical_or`/`logical_not`/`where` chained as another numpy
call's argument, nested directly or via an intermediate variable, including
more than one level of chaining; a chain past the supported depth declines
explicitly instead of misreading. See `regression/numpy/chaining_*`).

---

## Community testing readiness

ESBMC's standard across every frontend (C, C++, Solidity, Java/Kotlin) is
sound-but-incomplete, not full language/library coverage: whatever falls
outside the currently supported subset must reject with an explicit
diagnostic (ADR-NP principle 3) rather than silently return a wrong
verdict. By that bar, every gap in "Missing indexing / slicing" and
"Missing API surface" above is **not** a blocker for community testing —
each one already rejects explicitly instead of misbehaving.

With the call-result chaining fix above, there are no known soundness bugs
left in this file. **A build can be cut for community testing at any point
from here** — everything remaining is documented backlog that surfaces as
an explicit "not supported yet" diagnostic, not a wrong answer.

---

## Prioritised next steps

Nothing below blocks community testing (see above); this is post-release
backlog, in priority order:

1. **3-D+ and symbolic view descriptors (ADR-NP-003 etapa 3)** — extend the
   fixed-shape rank 1/2 descriptor model to higher ranks, symbolic
   shapes/axes/bounds, and broader stride combinations.
2. **General array returns** — consolidate assignment/type inference so user
   functions can return non-trivial NumPy arrays and sub-arrays safely.
3. **Remaining array method forms** — design plain ndarray `tolist`/`any`/`all`
   plus `a.sort()` and `a.argsort()` individually.
4. **Symbolic and broader multi-axis slicing** — support cases beyond the
   literal/fixed-shape recuts.
5. **Advanced dtype and constructor parity** — structured/object/custom dtype
   policy, diagnostics, and propagation.
6. **Random and iteration depth** — probability/replacement `choice`, extra
   distributions, and advanced `nditer`.
7. **Linear algebra breadth** — larger matrices, symbolic entries, and more
   faithful `norm`/`eig`/`svd`.

---

## Suggested next PRs

Each roadmap item above groups several sub-efforts; sizing them 1 PR per
item undercounts the real work. The two most recent NumPy PRs (arange
small-range performance, and this call-chaining fix) each took a small,
isolated gap through several commits of core fix plus a matching review
round; items below with multiple named consumers or distinct designs are
sized accordingly instead of assumed to be one PR each.

1. **3-D+ / symbolic view descriptors** (~2 PRs) — extend the rank 1/2
   fixed-shape descriptor model to higher ranks, symbolic axes/bounds/shapes,
   and broader non-literal stride combinations.
2. **General array returns** (~2 PRs) — root-cause the shared
   assignment/type-selection issue (two prior attempts already hit it)
   before extending to broader return support.
3. **Remaining array method forms** (~2 PRs) — plain ndarray
   `tolist`/`any`/`all` separately from `sort`/`argsort` (in-place vs copy
   semantics).
4. **Advanced dtype and constructors** (~2 PRs) — dtype policy
   (object/structured/custom) separate from constructor
   diagnostics/propagation.
5. **Random and iteration depth** (~2 PRs) — new distributions/`choice`
   separate from advanced `nditer`.
6. **Linear algebra expansion** (~2 PRs) — larger/symbolic matrix support
   separate from fuller `eig`/`svd`/`norm`.

**Total to close every item in this file: ~12 PRs.**

---

## Out of scope

- True SMT-array scalability beyond the current `array_typet` lowering; see
  ADR-NP-004.
- Extending the runtime-list model to hold array-typed elements; this remains
  disproportionately risky for current NumPy goals.
