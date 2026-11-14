import math

x = nondet_float()
if x >= 1.0 and x <= 2.0:
    assert math.gamma(x) > 0.89
