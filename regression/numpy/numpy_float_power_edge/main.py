import math
import numpy as np

r = np.float_power(np.array([0.0, 0.0, -2.0, 10.0, -8.0]), np.array([0.0, -1.0, 3.0, 400.0, 0.5]))
assert r[0] == 1.0 and r[1] == math.inf and r[2] == -8.0
assert r[3] == math.inf and r[4] != r[4]
assert np.float_power(0, 0) == 1.0
