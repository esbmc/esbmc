import math

x = nondet_float()
if x >= 1.0 and x <= 2.0:
    assert math.atan2(x, 1.0) < 0.5, "bound"
