# ESBMC NumPy — Remaining Work

**Updated:** 2026-08-14.

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
5. **Chaining one numpy call as another's argument is broken for
   non-constructor sources.** `np.logical_not(np.equal(a, b))` and
   `np.where(np.greater(a, b), x, y)` fail even when every operand is a
   plain literal list (no `arange`/constructor involved), both nested
   directly and through an intermediate variable — confirmed by direct
   execution. The comparison/logical dispatch block only knows how to
   resolve a `Name` argument back to a *constructor* call (`zeros`, `full`,
   `arange`, ...); a `Name` whose declaration is itself a call to another
   numpy function (`equal`, `greater`, ...) falls through to the old blind
   `args[0]` read, silently substituting that inner call's first argument
   for its actual (unevaluated) result. Found while extending
   `np.greater`/`np.equal` to accept `np.arange(...)` directly as an
   argument; those two now work — this is a distinct, pre-existing gap in
   general call composition, not arange-specific.

---

## Prioritised next steps

1. **General numpy call composition** — a `Name` argument whose declaration
   is itself a call to another numpy function (not a constructor) resolves
   incorrectly; root-cause and fix the shared resolve_var pattern instead of
   special-casing each affected dispatch block again.
2. **Definitive view descriptor model (ADR-NP-003 etapa 2)** — wire shared
   `buffer_id`/`offset`/`strides` metadata into reads, writes, views, and
   escape handling.
3. **General array returns** — consolidate assignment/type inference so user
   functions can return non-trivial NumPy arrays and sub-arrays safely.
4. **Remaining array method forms** — design `a.sort()`, `a.argsort()`,
   `a.tolist()`, `a.any()`, and `a.all()` individually.
5. **Symbolic and broader multi-axis slicing** — support cases beyond the
   literal/fixed-shape recuts.
6. **Advanced dtype and constructor parity** — structured/object/custom dtype
   policy, diagnostics, and propagation.
7. **Random and iteration depth** — probability/replacement `choice`, extra
   distributions, writable `nditer`, and real `.flat` mutation.
8. **Linear algebra breadth** — larger matrices, symbolic entries, and more
   faithful `norm`/`eig`/`svd`.

---

## Suggested next PRs

1. **Fix numpy call-result chaining** — `np.logical_not(np.equal(...))`,
   `np.where(np.greater(...), ...)`, and similar compositions, both nested
   and through an intermediate variable; small and isolated, currently the
   first prioritized roadmap item.
2. **Shared-buffer view descriptors (ADR-NP-003 etapa 2)** — unblock writable
   views, writable `.flat`, and writable `nditer`.
3. **General array returns** — fix the shared assignment/type-selection root
   cause before adding broader return support.
4. **Remaining array method forms** — implement method-specific designs for
   sort/argsort/tolist/any/all.
5. **Advanced dtype and constructors** — define dtype policy and improve
   constructor diagnostics/propagation.
6. **Random and iteration depth** — extend random/choice and iteration
   semantics.
7. **Linear algebra expansion** — extend matrix-size and symbolic-entry
   policies.

---

## Out of scope

- True SMT-array scalability beyond the current `array_typet` lowering; see
  ADR-NP-004.
- Extending the runtime-list model to hold array-typed elements; this remains
  disproportionately risky for current NumPy goals.
