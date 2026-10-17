import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
c = np.copy(v)
d = v.copy()
assert c[1] == 4
assert d[2] == 1
assert len(c) == 3
