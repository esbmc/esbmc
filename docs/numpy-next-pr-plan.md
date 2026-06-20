# NumPy Next PR Plan

This file groups the remaining NumPy roadmap into the smallest set of pull
requests that is still realistic to review and test independently.

Final review: PR 1 landed in the current branch and PR 2 remains the next
substantial unit. Keep them separate; merging PR 2 into a follow-up would make
the review surface too large and would mix unrelated failure modes.

## PR 2: NumPy math, linear algebra, and broad API coverage

### Scope
- Either model correctly or make explicit unsupported errors for:
  - `copysign`
  - `fmax`
  - `fmin`
  - `rint`
  - `remainder`
  - `nextafter`
  - `modf`
  - `frexp`
  - `isclose`
  - `np.e`
- Tighten integer `dot`/`matmul` overflow semantics to match the intended
  dtype model
- Add reductions:
  - `sum`, `prod`, `min`, `max`, `mean`, `argmin`, `argmax`
- Add comparison and logical ufuncs:
  - `>`, `<`, `>=`, `==`, `!=`
  - `logical_and`, `logical_or`, `logical_not`
- Add constructors:
  - `arange`, `full`, `eye`, `identity`, `linspace`
- Add general slicing and selection:
  - 1-D/2-D slicing
  - `where`
  - bounded boolean-mask indexing
- Add remaining common math functions:
  - `tan`
  - `arcsin`
  - `log`
  - `log2`
  - `log10`
  - `sinh`
  - `cosh`
  - `tanh`

### Why this PR exists
- These are currently a silent-wrong risk
- Integer matrix ops still have a correctness gap
- These are the most broadly useful missing array primitives
- They build on the corrected arithmetic and folding substrate from PR 1
- Slicing and selection are the next biggest gaps for realistic NumPy code
- The remaining scalar math set is straightforward once the core paths exist

### Suggested split inside the PR
- Batch 1: `copysign`, `fmax`, `fmin`
- Batch 2: `rint`, `remainder`, `nextafter`
- Batch 3: `modf`, `frexp`, `isclose`, `np.e`
- Keep `dot`/`matmul` overflow handling in the same PR only if it stays small;
  otherwise split it back out

### Tests
- Success: each fixed function behaves as expected
- Fail: each unsupported function rejects with the exact diagnostic
- Fail: overflow witness exercises the chosen `dot` boundary
- Edge: signed zero, NaN, dtype-width boundaries, and mixed-size inputs
- Success: each reduction/comparison/constructor has a literal happy path
- Fail: unsupported shape or empty-array boundaries fail cleanly
- Edge: axis handling, boolean outputs, and dtype behavior

### Docs
- Remove each fixed math function from the KNOWNBUG list
- Move any still-unsupported function into the explicit-unsupported section
- Update the linear algebra table
- Clarify the final overflow contract
- Update the coverage matrix for reductions, comparisons, constructors, linear
  algebra, slicing, selection, and the remaining math functions
- Remove any no-longer-true "Missing" entries

## Notes for every PR

- Each PR must include success, fail, and edge regression tests
- Keep at least one passing and one failing regression test
- Prefer small, isolated commits
- Update `docs/numpy-support-assessment.md` in the same PR when a limitation is
  removed
- Do not expand the 2-D ceiling unless the PR explicitly targets that work

## Commit order

### PR 2 commits
1. Clean up the NumPy math `KNOWNBUG` functions
2. Tighten integer `dot`/`matmul` overflow semantics
3. Add reductions, comparisons, logical ufuncs, and constructors
4. Add slicing, `where`, bounded mask indexing, and the remaining common math
