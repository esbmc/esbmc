import math

assert math.isclose(1.0, 1.0)
assert math.isclose(1.0, 1.0000000001, rel_tol=1e-8)
