import numpy as np

r = np.reciprocal(np.array([1, 2, -1, -3]))
assert r[0] == 1 and r[1] == 0 and r[2] == -1 and r[3] == 0
assert r.dtype == np.int64
f = np.reciprocal(np.array([2.0, -4.0]))
assert f[0] == 0.5 and f[1] == -0.25
assert np.reciprocal(2) == 0
assert np.reciprocal(2.0) == 0.5
