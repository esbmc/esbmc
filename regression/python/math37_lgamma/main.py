import math

r = math.lgamma(5.0)
expected = math.log(24.0)
assert math.fabs(r - expected) < 1e-6
