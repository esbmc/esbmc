import math

r = math.ldexp(0.5, 4)
assert math.fabs(r - 8.0) < 1e-12
