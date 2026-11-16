import numpy as np

i = np.array([7, -7])
q = np.floor_divide(i, 2.0)
assert q[0] == 3.0 and q[1] == -4.0 and q.dtype == np.float64
m = np.remainder(i, 2.5)
assert m[0] == 2.0 and m[1] == 0.5
f = np.fmod(i, 2.5)
assert f[0] == 2.0 and f[1] == -2.0
a = np.fabs(np.remainder(i, -3))
assert a[0] == 2.0 and a[1] == 1.0 and a.dtype == np.float64
r = np.reciprocal(np.floor_divide(i, 2))
assert r[0] == 0 and r[1] == 0
p = np.float_power(np.fmod(i, 4), 2)
assert p[0] == 9.0 and p[1] == 9.0
