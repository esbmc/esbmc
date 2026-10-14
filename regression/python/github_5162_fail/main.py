# Negative variant of github_5162: len(A[0]) on a List[List[float]] parameter
# now computes the correct inner-list length, so an assertion that contradicts
# it must be reported as VERIFICATION FAILED rather than silently passing.
from typing import List


def k(A: List[List[float]]) -> int:
    w = len(A[0])
    out = [0.0 for _ in range(w)]
    return len(out)


A = [[1.0], [2.0]]
assert k(A) == 2
