import numpy as np

a = np.array([7, -7, 7, -7])
r = np.fmod(a, np.array([2, 2, -2, -2]))
assert r[0] == 1 and r[1] == -1 and r[2] == 1 and r[3] == -1
assert r.dtype == np.int64
f = np.fmod(np.array([5.5, -5.5]), -2.0)
assert f[0] == 1.5 and f[1] == -1.5
v = np.fmod(a[1:3], 3)
assert v[0] == -1 and v[1] == 1
assert np.fmod(-7, 2) == -1
