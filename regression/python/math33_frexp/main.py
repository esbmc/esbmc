import math

m, e = math.frexp(8.0)
assert math.fabs(m - 0.5) < 1e-12
assert e == 4
