# NumPy Next PR Plan

This file groups the remaining NumPy roadmap into the smallest set of pull
requests that is still realistic to review and test independently.

Current branch state: the current PR lands the reductions, comparison/logical
ufuncs, constructors, `where`, and the common scalar math family. Keep the
remaining work separate so review stays focused and the failure modes do not
get mixed.

## PR 2: NumPy overflow semantics

### Scope
- Tighten integer `dot`/`matmul` overflow semantics to match the intended
  dtype model

### Why this PR exists
- Integer matrix ops still have a correctness gap
- Overflow handling changes verdicts for realistic integer NumPy code
- This is the smallest remaining soundness-oriented NumPy change

### Tests
- Success: overflow-checked `dot`/`matmul` matches the intended dtype model
- Fail: overflow witnesses exercise the chosen `dot`/`matmul` boundary
- Edge: signed zero, NaN, dtype-width boundaries, and mixed-size inputs

### Docs
- Update the linear algebra table
- Clarify the final overflow contract

## PR 3: NumPy slicing and mask indexing

### Scope
- Add general slicing and selection:
  - 1-D/2-D slicing
  - bounded boolean-mask indexing
- Close any remaining follow-up around `where`/selection semantics if the
  bounded-mask model needs extra guardrails

### Why this PR exists
- Slicing and mask indexing remain the largest practical coverage gaps
- These are the remaining pieces that still change verdicts for realistic
  NumPy code in this area
- They build on the reductions/comparison/constructor/math coverage already
  landing in the current branch

### Tests
- Success: slicing and mask-indexing happy paths return the expected values
- Fail: unsupported slice/mask shapes reject with the exact diagnostic
- Edge: slice bounds, empty results, signed zero, NaN, and mixed-size inputs

### Docs
- Update the coverage matrix for slicing, selection, and the remaining
  indexing semantics
- Remove any no-longer-true "Missing" entries tied to the current branch work

## Notes for every PR

- Each PR must include success, fail, and edge regression tests
- Keep at least one passing and one failing regression test
- Prefer small, isolated commits
- Update `docs/numpy-support-assessment.md` in the same PR when a limitation is
  removed
- Do not expand the 2-D ceiling unless the PR explicitly targets that work

## Commit order

### PR 2 commits
1. Tighten integer `dot`/`matmul` overflow semantics

### PR 3 commits
1. Add 1-D/2-D slicing support
2. Add bounded boolean-mask indexing and `where` cleanup
