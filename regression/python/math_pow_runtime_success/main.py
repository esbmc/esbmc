import math

x = nondet_float()
y = nondet_float()
if x >= 1.0 and x <= 2.0 and y >= 0.0 and y <= 1.0:
    r = math.pow(x, 2.0)
    assert abs(r - x * x) <= 1e-12 * r
    q = math.pow(2.0, y)
    assert q >= 1.0 - 1e-12 and q <= 2.0 + 1e-12
