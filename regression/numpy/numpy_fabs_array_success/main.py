import numpy as np

a = np.array([-1.5, 2.0, -3.0])
r = np.fabs(a)
assert r[0] == 1.5 and r[1] == 2.0 and r[2] == 3.0
assert r.dtype == np.float64
i = np.fabs(np.array([-1, 2]))
assert i[0] == 1.0 and i[1] == 2.0
assert i.dtype == np.float64
v = np.fabs(a[1:])
assert v[0] == 2.0 and v[1] == 3.0
