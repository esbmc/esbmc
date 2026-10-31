import numpy as np

a = np.full(3, 2.5, dtype=np.int64)
c = a.copy()
a[0] = 9
assert c[0] == 2 and a[0] == 9

b = np.array([100, -100], dtype=np.int8)
d = b.copy()
d[0] += 100
assert d[0] == -56 and b[0] == 100 and d.itemsize == 1
