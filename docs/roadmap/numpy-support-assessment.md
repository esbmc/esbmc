# ESBMC NumPy — Remaining Work

**Updated:** 2026-08-16.

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
| Final shared-buffer view model | Missing | ADR-NP-003 etapa 1 is only a conservative guard layer. The runtime representation still copies many view-like operations. Etapa 2 must connect `ndarray_descriptor` `buffer_id`, `offset`, and `strides` to indexing, assignment, transpose, reshape/ravel, escape checks, `.flat`, and writable `nditer` so supported views alias like real NumPy. |
| Higher-dimensional or symbolic slice bounds beyond literal-copy cases | Missing | Literal/fixed-shape cases such as bounded 2-D column slices and one-/two-slice-axis mixed tuple indexing are supported. Three or more slice axes, symbolic slice bounds, non-literal strides, and broader stride combinations remain explicitly rejected. |
| `a.sort()` / `a.argsort()` / `a.tolist()` method forms | Missing | These are not plain method-to-module rewrites. `a.sort()` is in-place while `np.sort(a)` returns a copy; `a.argsort()` needs `np.argsort()` to support variables first; `a.tolist()` needs new array-to-list conversion logic. |
| `a.any()` / `a.all()` method forms | Missing, cause unclear | `a.any()` currently dispatches like Python builtin `any()` with no argument (`ERROR: any() expected at least 1 argument, got 0`). Root-cause before adding a rewrite, since it may indicate a broader method-dispatch issue. |

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
| Views / strides | Final shared-buffer alias semantics, writable views, offset/stride based assignment, and broad non-literal stride support. |
| Iteration | Writable `nditer`, advanced `op_flags`, multi-operand iteration, and mutation through `.flat`. |

---

## Soundness / performance concerns

1. **Constant-folding bypasses ESBMC's overflow/rounding checks** for folded
   paths. Use `--python-no-fold` to force SMT encoding and compare verdicts.
2. **Element-wise broadcasting** still requires concrete shapes at conversion
   time; symbolic shapes work only for selected array creation paths.
3. **Scalability wall** (#5121): arrays are still represented as fully
   unrolled value lists. Large arrays can explode even when the operation is
   conceptually simple.
4. **Basic-indexing views still do not alias at runtime.** Covered mutations
   are rejected conservatively, but the final ADR-NP-003 descriptor model is
   still needed to replace rejection with faithful aliasing.

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

1. **Definitive view descriptor model (ADR-NP-003 etapa 2)** — wire shared
   `buffer_id`/`offset`/`strides` metadata into reads, writes, views, and
   escape handling.
2. **General array returns** — consolidate assignment/type inference so user
   functions can return non-trivial NumPy arrays and sub-arrays safely.
3. **Remaining array method forms** — design `a.sort()`, `a.argsort()`,
   `a.tolist()`, `a.any()`, and `a.all()` individually.
4. **Symbolic and broader multi-axis slicing** — support cases beyond the
   literal/fixed-shape recuts.
5. **Advanced dtype and constructor parity** — structured/object/custom dtype
   policy, diagnostics, and propagation.
6. **Random and iteration depth** — probability/replacement `choice`, extra
   distributions, writable `nditer`, and real `.flat` mutation.
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

1. **Shared-buffer view descriptors (ADR-NP-003 etapa 2)** (~3 PRs) —
   `buffer_id`/`offset`/`strides` must reach 7 distinct consumers
   (indexing, assignment, transpose, reshape/ravel, escape checks, `.flat`,
   writable `nditer`); one PR risks becoming too large to review.
2. **General array returns** (~2 PRs) — root-cause the shared
   assignment/type-selection issue (two prior attempts already hit it)
   before extending to broader return support.
3. **Remaining array method forms** (~3 PRs) — `sort`/`argsort` (in-place vs
   copy semantics), `tolist` (new conversion logic), and `any`/`all` (root
   cause still unclear) are three distinct designs, not one.
4. **Advanced dtype and constructors** (~2 PRs) — dtype policy
   (object/structured/custom) separate from constructor
   diagnostics/propagation.
5. **Random and iteration depth** (~2 PRs) — new distributions/`choice`
   separate from writable `nditer`/`.flat` (which also depends on item 1).
6. **Linear algebra expansion** (~2 PRs) — larger/symbolic matrix support
   separate from fuller `eig`/`svd`/`norm`.

**Total to close every item in this file: ~14 PRs.**

---

## Out of scope

- True SMT-array scalability beyond the current `array_typet` lowering; see
  ADR-NP-004.
- Extending the runtime-list model to hold array-typed elements; this remains
  disproportionately risky for current NumPy goals.
