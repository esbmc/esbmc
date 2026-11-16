import math
import numpy as np

f = np.reciprocal(np.array([0.0, -0.0, math.inf]))
assert f[0] == math.inf and f[1] == -math.inf and f[2] == 0.0
r = np.reciprocal(np.array([0, 1]))
assert r[1] == 1
