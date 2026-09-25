# Negative counterpart: the lowered abs must reach the solver and report a real
# violation rather than being masked by the library model's result.
import math


def f(x: float) -> float:
    return math.fabs(x)


v = -3.5 if True else 1.0
assert f(v) == -3.5
