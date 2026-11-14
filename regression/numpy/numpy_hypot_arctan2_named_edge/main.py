import math
import numpy as np

a = np.array([-0.0, 1e300, 1.0])
b = np.array([-1.0, 1e-300, np.inf])
t = np.arctan2(a, b)
assert t[0] == -math.pi
assert t[2] == 0.0
h = np.hypot(a, b)
assert h[0] == 1.0
assert abs(h[1] - 1e300) <= 1e288
assert h[2] == np.inf
