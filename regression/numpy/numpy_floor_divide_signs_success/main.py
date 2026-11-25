import numpy as np

a = np.array([7, -7, 7, -7, 6])
b = np.array([2, 2, -2, -2, 3])
q = np.floor_divide(a, b)
assert q[0] == 3 and q[1] == -4 and q[2] == -4 and q[3] == 3 and q[4] == 2
assert q.dtype == np.int64
assert np.floor_divide(-7, 2) == -4
f = np.floor_divide(np.array([7.5, -7.5]), 2.0)
assert f[0] == 3.0 and f[1] == -4.0
assert f.dtype == np.float64
d = np.floor_divide(np.array([1.0, -1.0]), np.array([0.1, 0.1]))
assert d[0] == 9.0 and d[1] == -10.0
