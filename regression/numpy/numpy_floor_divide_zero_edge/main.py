import math
import numpy as np

q = np.floor_divide(np.array([7, -7, -9223372036854775807 - 1]), np.array([0, 0, -1]))
assert q[0] == 0 and q[1] == 0 and q[2] == -9223372036854775807 - 1
r = np.remainder(np.array([7, -9223372036854775807 - 1]), np.array([0, -1]))
assert r[0] == 0 and r[1] == 0
f = np.floor_divide(np.array([1.0, -1.0, 0.0, -1.0]), np.array([0.0, 0.0, 0.0, math.inf]))
assert f[0] == math.inf and f[1] == -math.inf and f[2] != f[2] and f[3] == -1.0
m = np.remainder(np.array([1.0, 0.0, 1.0, math.nan]), np.array([0.0, -3.0, -math.inf, 2.0]))
assert m[0] != m[0] and m[2] == -math.inf and m[3] != m[3]
assert m[1] == 0.0 and math.copysign(1.0, m[1]) == -1.0
