# Negative counterpart: the lowered ieee_sqrt must reach the solver and report a
# real violation, not be masked by a NaN result that compares false against
# everything.
import math


def root(n: int) -> float:
    return math.sqrt(n)


x = 9 if True else 4
assert root(x) == 4.0
