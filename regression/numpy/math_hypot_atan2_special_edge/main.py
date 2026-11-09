import math
import numpy as np

inf = float("inf")
nan = float("nan")
assert np.hypot(inf, nan) == inf and math.hypot(nan, -inf) == inf
r = np.hypot(nan, 1.0)
assert r != r
assert np.hypot(0.0, -3.0) == 3.0 and math.hypot(-2.0, 0.0) == 2.0
assert np.arctan2(0.0, -0.0) == math.pi and math.atan2(-0.0, -1.0) == -math.pi
assert np.arctan2(1.0, 0.0) == math.pi / 2
assert math.atan2(inf, inf) == math.pi / 4
z = np.arctan2(-0.0, 1.0)
assert z == 0.0 and math.copysign(1.0, z) == -1.0
r = math.atan2(nan, 1.0)
assert r != r

y = nondet_float()
if y > 0.0 and y < inf:
    assert np.arctan2(y, inf) == 0.0 and np.hypot(y, 0.0) == y
