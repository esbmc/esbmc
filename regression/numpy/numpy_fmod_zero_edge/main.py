import math
import numpy as np

r = np.fmod(np.array([7, -7, -9223372036854775807 - 1]), np.array([0, 0, -1]))
assert r[0] == 0 and r[1] == 0 and r[2] == 0
f = np.fmod(np.array([1.0, -4.0, 1.0, math.inf, -0.0]), np.array([0.0, 2.0, -math.inf, 2.0, 3.0]))
assert f[0] != f[0] and f[2] == 1.0 and f[3] != f[3]
assert f[1] == 0.0 and math.copysign(1.0, f[1]) == -1.0
assert f[4] == 0.0 and math.copysign(1.0, f[4]) == -1.0
