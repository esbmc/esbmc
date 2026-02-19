import math

r = math.erfc(1.0)
assert math.fabs(r - 0.15729921) < 1e-6
