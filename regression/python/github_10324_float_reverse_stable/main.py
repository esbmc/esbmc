import math


def desc(xs: list[float]) -> list[float]:
    return sorted(xs, reverse=True)


def asc(xs: list[float]) -> list[float]:
    return sorted(xs)


s = desc([-0.0, 0.0])
assert math.copysign(1.0, s[0]) == -1.0
assert math.copysign(1.0, s[1]) == 1.0
t = asc([0.0, -0.0])
assert math.copysign(1.0, t[0]) == 1.0
u = asc([-1.0, -2.0, 0.5])
assert u[0] == -2.0 and u[1] == -1.0 and u[2] == 0.5
