import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
assert v.tolist() == [1, 4, 1]
c = v.copy()
assert c[1] == 4
f = v.flatten()
assert f[2] == 1
r = v.ravel()
r[0] = 6
assert a[1] == 6
