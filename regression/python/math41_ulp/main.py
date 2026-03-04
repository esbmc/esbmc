import math

u = math.ulp(1.0)
expected = math.pow(2.0, -52.0)
assert u > 0.0
assert math.fabs(u - expected) < 1e-12
