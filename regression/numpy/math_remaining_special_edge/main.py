import math

inf = float("inf")
assert math.expm1(-inf) == -1.0
assert math.log1p(inf) == inf
assert math.cbrt(-inf) == -inf
assert math.acosh(inf) == inf
r = math.atanh(float("nan"))
assert r != r
assert math.copysign(1.0, math.log1p(-0.0)) == -1.0
