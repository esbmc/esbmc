# NumPy Next PR Plan

This file is meant to be executable by another LLM with minimal context
gathering. It describes the remaining NumPy work, the exact files that should
change in each commit, the tests that must be added, and the docs that need to
be updated.

The current branch already covers:

- reductions: `sum`, `prod`, `min`, `max`, `mean`, `argmin`, `argmax`
- comparison/logical ufuncs: `greater`, `less`, `equal`, `logical_and`,
  `logical_or`, `logical_not`
- constructors: `array`, `zeros`, `ones`, `full`, `arange`, `eye`,
  `identity`, `linspace`
- `where`
- common math: `sin`, `cos`, `tan`, `arcsin`, `arctan`, `log`, `log2`,
  `log10`, `sinh`, `cosh`, `tanh`, `sqrt`, `exp`, `fabs`, `ceil`, `floor`,
  `round`, `trunc`, `arccos`, `copysign`, `fmax`, `fmin`, `rint`, `modf`,
  `frexp`, `remainder`, `nextafter`, `isclose`
- `dot`, `matmul`, `transpose`, `det` at the currently supported boundaries

The remaining work is one PR:

1. ~~integer overflow semantics for `dot` / `matmul`~~ — **DONE** (PR 2 landed)
2. general slicing and bounded boolean-mask indexing

Only slicing remains.

---

## Shared implementation rules

Before editing code, keep these constraints in mind:

- stay within the current 2-D array ceiling
- keep literal-array lowering behavior deterministic
- do not expand the scope to fancy indexing, reshape, random, or structured
  arrays
- every PR must include:
  - at least one success regression
  - at least one fail regression
  - at least one edge regression
- update `docs/numpy-support-assessment.md` in the same PR when a limitation is
  removed
- keep changes small enough that each commit has one clear purpose

---

## PR 2: integer `dot` / `matmul` overflow semantics

### Goal

Fix the remaining integer overflow behavior for `np.dot` and `np.matmul`
without changing the supported shapes or adding new array semantics.

This PR is only about integer accumulation / wrapping behavior. It should not
touch slicing, `where`, constructors, or generic NumPy math.

### What should be implemented

- integer accumulation must not silently rely on host `int64_t` widening
- the result should respect the target dtype width
- signed and unsigned integer paths must stay explicit
- float behavior must remain unchanged
- literal 1-D and 2-D inputs should still follow the same lowering path
- runtime-backed `dot` / `matmul` behavior should stay aligned with the frontend
  contract

### Code path to inspect first

The next contributor should start by reading these files:

- `src/python-frontend/numpy_call_expr.cpp`
  - this is the main lowering entry for `np.dot` and `np.matmul`
  - it decides whether a call folds, lowers to the OM, or rejects the input
- `src/c2goto/library/python/linalg.c`
  - this is the runtime OM used by the frontend for linear algebra cases
- `regression/numpy/roadmap_api_*`
  - these already cover neighboring matrix behaviors and are good canaries for
    side effects

### Semantics to preserve

- keep the supported shape set unchanged
- do not alter the existing error text for unsupported ranks or shapes unless
  the overflow fix requires a new explicit diagnostic
- do not change the float path, only the integer path
- keep the behavior of literal arrays and runtime arrays aligned
- keep boolean inputs stable if they already route through the current path

### Commit 1: tighten the lowering and accumulation path

#### Suggested commit message

- `[python-frontend] tighten numpy dot and matmul overflow handling`

#### Files to modify

- `src/python-frontend/numpy_call_expr.cpp`
  - adjust `dot` / `matmul` dispatch and lowering
  - make integer overflow handling explicit
  - keep the float path untouched
- `src/c2goto/library/python/linalg.c`
  - adjust helper behavior only if the runtime OM needs to match the new
    frontend contract

If `dot` and `matmul` stop sharing the exact same lowering path, split this
into two code commits:

- one commit for `dot`
- one commit for `matmul`

If the same patch covers both cleanly, keep it as one commit.

If a code change affects only `dot` or only `matmul`, keep it as a
single-operation commit.

#### Behavior expected after this commit

- integer `dot` / `matmul` should no longer depend on a widened host
  accumulator for the final answer
- literal and runtime-backed integer paths should agree on the result
- float inputs should still follow the existing path without semantic changes
- the supported rank/shape matrix should remain exactly what it is today
- if a code change affects only `dot` or only `matmul`, split it into a
  single-operation commit

#### Implementation checklist

- inspect the current accumulator width in the integer branch
- confirm whether the result is currently widened to `int64_t` implicitly
- add an explicit overflow guard or exact-width wrap before returning the
  result
- keep the promotion rule for mixed int/float inputs unchanged
- make sure 1-D x 1-D and 2-D x 2-D still go through the same top-level
  entrypoints
- if the runtime OM needs a helper adjustment, keep it local to `linalg.c`
- avoid touching transpose or det logic
- avoid introducing new array constructors or new rank handling

### Commit 2: add success regressions for dot and matmul

#### Suggested commit message

- `test: add numpy dot and matmul success regressions`

#### Files to add

- `regression/numpy/dot_overflow_success/main.py`
- `regression/numpy/dot_overflow_success/test.desc`
- `regression/numpy/matmul_overflow_success/main.py`
- `regression/numpy/matmul_overflow_success/test.desc`

#### What these tests must prove

- small integer arrays verify exactly
- `dot` and `matmul` still return the expected result on non-overflowing inputs
- only literal nested lists are used
- the result shape is what the model already supports
- the verdict for these regressions should be `VERIFICATION SUCCESSFUL`
- each file should cover exactly one behavior family

#### Test content requirements

- 1-D x 1-D or 2-D x 2-D depending on the operation
- no overflow boundary values
- exact scalar or nested-list equality
- keep success files free of any intended-failure condition

### Commit 3: add failure regressions for dot and matmul

#### Suggested commit message

- `test: add numpy dot and matmul failure regressions`

#### Files to add

- `regression/numpy/dot_overflow_fail/main.py`
- `regression/numpy/dot_overflow_fail/test.desc`
- `regression/numpy/matmul_overflow_fail/main.py`
- `regression/numpy/matmul_overflow_fail/test.desc`

#### What these tests must prove

- an overflow boundary is actually exercised
- the modeled verdict is the intended one
- the failure is explicit and not masked by an unrelated shape error
- the verdict for these regressions should be `VERIFICATION FAILED`
- each file should contain only one overflow witness

#### Test content requirements

- values that cross the integer dtype boundary
- a single clear witness per file
- small matrices only
- keep shape errors out of these files so the failure is clearly about
  overflow

### Commit 4: add edge regressions for dot and matmul

#### Suggested commit message

- `test: add numpy dot and matmul edge regressions`

#### Files to add

- `regression/numpy/dot_overflow_edge/main.py`
- `regression/numpy/dot_overflow_edge/test.desc`
- `regression/numpy/matmul_overflow_edge/main.py`
- `regression/numpy/matmul_overflow_edge/test.desc`

#### What these tests must prove

- boundary-adjacent values are handled consistently
- one-step-below / one-step-above cases are covered
- zero-adjacent or signed-extreme values are represented where relevant
- the verdict for these regressions should match the modeled boundary case
- do not combine a shape failure with an overflow edge in the same file

#### Test content requirements

- include both maximum and minimum edges when signed integers are used
- keep the inputs literal and small
- prefer one case per file so failures are easy to interpret

### Commit 5: update the support assessment and remove the finished gap from the plan

#### Suggested commit message

- `docs: update numpy support assessment for dot and matmul overflow`

#### Files to modify

- `docs/numpy-support-assessment.md`
  - update the `dot` and `matmul` rows
  - remove the overflow caveat if it is fully fixed
  - keep the 2-D ceiling note only if it still applies
- `docs/numpy-next-pr-plan.md`
  - remove the overflow work from the remaining plan
  - leave slicing as the only open follow-up

#### What the docs should say after this commit

- `dot` / `matmul` should no longer be described as having an unresolved
  integer overflow gap
- the remaining limitation list should still mention:
  - general slicing
  - bounded boolean-mask indexing
  - any still-missing high-level NumPy APIs
- the support matrix should still call out any true residual limitation in
  rank support or dtype conversion, but not overflow if it is fixed
- the remaining plan should read as “slicing only remains” after this PR lands

### PR 2 validation checklist

- `ctest -L numpy -j8 --timeout 120 --output-on-failure`
- the six new `dot` / `matmul` regressions pass
- existing `numpy` regressions still pass
- `docs/numpy-support-assessment.md` matches the actual behavior
- no `github`, `math`, or `complex` lateral regression appears in the existing
  suite selection used by the branch
- a commit is complete only when its isolated regressions pass
- if `dot` and `matmul` are split, each code commit must pass independently

#### Minimal test matrix for each regression directory

Each `main.py` should do one thing only:

- import `numpy as np`
- create literal arrays
- call exactly one target function family (`dot` or `matmul`)
- assert the expected scalar or nested-list result
- keep the file short enough to understand without context

Each `test.desc` should follow the existing format:

- line 1: `CORE`
- line 2: `main.py`
- line 3: the exact ESBMC flags already used by similar NumPy regressions
- line 4+: the exact expected output regex

---

## PR 3: general slicing and bounded boolean-mask indexing

### Goal

Add the remaining practical indexing coverage:

- 1-D and 2-D slicing
- bounded boolean-mask indexing

Do not add fancy indexing or any 3-D lifting in this PR.

### What should be implemented

- `a[i:j]` on 1-D arrays
- `a[:, j]`, `a[i, :]`, and other bounded 2-D slice forms that fit the current
  model
- boolean-mask selection for 1-D and 2-D literal arrays
- clear rejection for unsupported slice steps, shapes, and mask dimensions
- do not touch `where` unless the new mask path proves a real bug

### Code path to inspect first

The next contributor should start here:

- `src/python-frontend/numpy_call_expr.cpp`
  - slice and mask patterns originate here
- `src/python-frontend/python_list.cpp`
  - this is where list-backed selection and nested-list extraction already live
- `src/c2goto/library/python/slice.c`
  - use this if runtime slice behavior needs to be extended
- `src/c2goto/library/python/list.c`
  - use this if the selection result must be produced by the list OM
- `src/python-frontend/models/numpy.py`
  - update only if the type-inference stub needs a return-shape hint

### Commit 1: add slice lowering and extraction support

#### Suggested commit message

- `[python-frontend] add bounded numpy slicing support`

#### Files to modify

- `src/python-frontend/numpy_call_expr.cpp`
  - recognize slice-like NumPy constructs that should lower into the existing
    array/list model
  - keep the literal-array path explicit
- `src/python-frontend/python_list.cpp`
  - add or extend helper logic for extracting slice results if needed
  - preserve the current nested-list behavior for plain indexing
- `src/c2goto/library/python/slice.c`
  - extend the slice operational model only as far as needed for the
    regressions
- `src/c2goto/library/python/list.c`
  - adjust list helper behavior if slice extraction reuses runtime list OMs

If 1-D slicing and 2-D slicing end up needing different lowering code, split
them into separate commits:

- one commit for 1-D slicing
- one commit for 2-D slicing

Keep them together only if the same helper handles both cleanly.

If one of the slice commits only affects 1-D or only affects 2-D, keep it as a
single-purpose commit.

#### Behavior expected after this commit

- literal 1-D slices should return the exact selected sublist
- literal 2-D slices should return the exact selected submatrix
- empty slices should stay empty and not crash or widen
- unsupported steps or shapes should fail loudly with the existing style of
  diagnostics
- plain indexing must remain unchanged

### Commit 2: add slice success regressions

#### Suggested commit message

- `test: add numpy slicing success regressions`

#### Files to add

- `regression/numpy/slice_1d_success/main.py`
- `regression/numpy/slice_1d_success/test.desc`
- `regression/numpy/slice_2d_success/main.py`
- `regression/numpy/slice_2d_success/test.desc`

#### What these tests must prove

- a middle 1-D slice works
- a 2-D row/column slice works
- the returned value and shape are exact
- the verdict for these regressions should be `VERIFICATION SUCCESSFUL`
- each file should cover exactly one slice family

#### Test content requirements

- one literal input array
- one slicing expression
- one expected result
- no unrelated NumPy helper calls
- keep success cases free of failure-only guards

### Commit 3: add slice failure regressions

#### Suggested commit message

- `test: add numpy slicing failure regressions`

#### Files to add

- `regression/numpy/slice_fail/main.py`
- `regression/numpy/slice_fail/test.desc`

#### What these tests must prove

- zero-step slices fail
- unsupported slice shapes fail
- rank mismatches fail explicitly
- the verdict for these regressions should be `VERIFICATION FAILED`
- do not mix multiple failure causes in the same file

#### Test content requirements

- one clear failure witness per file
- small literal arrays only
- explicit diagnostic in the expected output regex
- keep one reason for failure per file

### Commit 4: add slice edge regressions

#### Suggested commit message

- `test: add numpy slicing edge regressions`

#### Files to add

- `regression/numpy/slice_edge/main.py`
- `regression/numpy/slice_edge/test.desc`

#### What these tests must prove

- empty slices are handled cleanly
- first-element and last-element boundaries work
- the empty 2-D row/column path, if supported, is stable
- the verdict for these regressions should match the modeled edge behavior
- each file should stay within one edge family

#### Test content requirements

- one edge case per file
- keep the inputs literal and tiny
- prefer one assertion only
- do not mix edge behavior with shape mismatches in the same file

### Commit 5: add bounded boolean-mask indexing

#### Suggested commit message

- `[python-frontend] add bounded numpy boolean mask indexing`

#### Files to modify

- `src/python-frontend/numpy_call_expr.cpp`
  - detect and lower the mask-indexing pattern
- `src/python-frontend/python_list.cpp`
  - materialize the selected elements if the list model is reused
- `src/c2goto/library/python/list.c`
  - extend the runtime list support only if needed for mask filtering
- `src/python-frontend/models/numpy.py`
  - keep the type-inference stub aligned with the new behavior if a stub change
    is needed

If the mask detection and the selected-element materialization turn into
separate patches, split them:

- one commit for detection / lowering
- one commit for materialization / runtime selection

Keep them together only if the path is identical enough to stay readable.

#### Behavior expected after this commit

- a mask that matches the leading dimension should select the correct values
- result order must follow the source order
- all-true, all-false, and mixed masks should be handled deterministically
- invalid shapes should fail clearly instead of being misinterpreted as slices
- existing `where` behavior should not change unless a bug is discovered in the
  shared selection logic

#### Implementation checklist

- bounded mask selection on 1-D and 2-D arrays
- explicit rejection for invalid mask length or shape
- keep `where` unchanged unless the new mask path reveals a follow-up bug
- make the mask selection result deterministic for literal arrays
- keep the selected element order stable and match the original list order
- do not introduce a generalized fancy-indexing subsystem
- if a helper is shared with `where`, keep the special case narrow enough that
  it does not alter unrelated logical ufunc behavior

### Commit 6: add mask success regressions

#### Suggested commit message

- `test: add numpy boolean mask success regressions`

#### Files to add

- `regression/numpy/bool_mask_index_success/main.py`
- `regression/numpy/bool_mask_index_success/test.desc`

#### What these tests must prove

- mixed selection works
- all-true selection works
- result order matches the source order
- the verdict for these regressions should be `VERIFICATION SUCCESSFUL`
- each file should cover exactly one mask family

#### Test content requirements

- small literal arrays only
- masks that match the leading dimension
- exact selected result
- keep one mask shape per file

### Commit 7: add mask failure regressions

#### Suggested commit message

- `test: add numpy boolean mask failure regressions`

#### Files to add

- `regression/numpy/bool_mask_index_fail/main.py`
- `regression/numpy/bool_mask_index_fail/test.desc`

#### What these tests must prove

- mask length mismatch fails
- mask rank mismatch fails
- unsupported mask shapes fail clearly
- the verdict for these regressions should be `VERIFICATION FAILED`
- do not mix mask length and mask rank failures in the same file

#### Test content requirements

- one clear failure witness per file
- explicit diagnostic in the expected output regex
- keep one reason for failure per file

### Commit 8: add mask edge regressions

#### Suggested commit message

- `test: add numpy boolean mask edge regressions`

#### Files to add

- `regression/numpy/bool_mask_index_edge/main.py`
- `regression/numpy/bool_mask_index_edge/test.desc`

#### What these tests must prove

- mask selects none
- empty output result is stable
- all-false edge cases remain deterministic
- the verdict for these regressions should match the modeled edge behavior
- each file should stay within one edge family

#### Test content requirements

- one edge case per file
- tiny literal arrays only
- one assertion only
- do not mix empty-result edges with shape-mismatch failures

### Commit 9: docs refresh for slicing and mask indexing

#### Suggested commit message

- `docs: update numpy slicing and mask coverage`

#### Files to modify

- `docs/numpy-support-assessment.md`
  - mark the new slice/mask subset as supported
  - keep any remaining fancy-indexing or 3-D limitations
- `docs/numpy-next-pr-plan.md`
  - remove the slicing PR from the remaining work once the code lands
  - leave only the true future gaps

#### What the docs should say after this commit

- slicing is supported only in the bounded 1-D/2-D subset implemented here
- boolean-mask selection is supported only for the bounded cases above
- fancy indexing remains out of scope
- 3-D lifting remains out of scope
- any remaining index gaps should be listed as future work, not unresolved
  ambiguity

### PR 3 validation checklist

- `ctest -L numpy -j8 --timeout 120 --output-on-failure`
- the new slice and mask regressions pass
- fail cases reject unsupported inputs cleanly
- docs match the actual implementation state
- any previously passing `numpy` regressions still pass after the new index
  lowering path is added
- do not touch `where` unless the new mask path proves a real bug
- a commit is complete only when its isolated regressions pass

---

## Commit ordering summary

If you are implementing these PRs, use this order:

### PR 2
1. code change for `dot` / `matmul` overflow
2. success regressions for `dot` / `matmul`
3. failure regressions for `dot` / `matmul`
4. edge regressions for `dot` / `matmul`
5. docs update

### PR 3
1. slice lowering
2. slice success regressions
3. slice failure regressions
4. slice edge regressions
5. boolean-mask indexing
6. mask success regressions
7. mask failure regressions
8. mask edge regressions
9. docs update

### Suggested commit messages at a glance

- PR 2 code: `[python-frontend] tighten numpy dot and matmul overflow handling`
- PR 2 docs: `docs: update numpy support assessment for dot and matmul overflow`
- PR 3 slice code: `[python-frontend] add bounded numpy slicing support`
- PR 3 slice tests: `test: add numpy slicing success regressions`
- PR 3 slice tests: `test: add numpy slicing failure regressions`
- PR 3 slice tests: `test: add numpy slicing edge regressions`
- PR 3 mask code: `[python-frontend] add bounded numpy boolean mask indexing`
- PR 3 mask tests: `test: add numpy boolean mask success regressions`
- PR 3 mask tests: `test: add numpy boolean mask failure regressions`
- PR 3 mask tests: `test: add numpy boolean mask edge regressions`
- PR 3 docs: `docs: update numpy slicing and mask coverage`

Keep the commits narrowly scoped. If a commit touches both code and tests, it
should still only cover one behavior family.

If a commit starts to mix two unrelated changes, split it before implementing.

---

## What remains after these PRs

If more NumPy work is needed later, the next candidates are:

- `reshape`
- `ravel`
- `flatten`
- `astype`
- fancy integer indexing
- statistics helpers
- `linalg.inv`, `solve`, `eig`, `svd`
- structured dtypes
- views / strides
- random helpers

Do not include those in PR 2 or PR 3 unless the scope is intentionally being
replanned.
