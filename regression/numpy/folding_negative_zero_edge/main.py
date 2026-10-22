import numpy as np

p = np.subtract(0.0, 0.0)
assert np.copysign(1.0, p) == 1.0
n = np.multiply(-1.0, 0.0)
assert n == 0.0
assert np.copysign(1.0, n) == -1.0
