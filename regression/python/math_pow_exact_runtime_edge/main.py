import math

x = nondet_float()
if x == 3.0:
    assert math.pow(x, 2.0) == 9.0
    assert math.pow(x, 3.0) == 27.0
y = nondet_float()
if y >= 1.0 and y <= 2.0:
    r = math.pow(y, 2.0)
    assert r >= 1.0 - 1e-12 and r <= 4.0 + 1e-12
