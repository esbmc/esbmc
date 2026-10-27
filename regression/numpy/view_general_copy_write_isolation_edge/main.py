import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
c = v.copy()
c[0] = 100
assert sum(v) == 6
v[1] = 50
assert sum(c) == 105
assert sum(v) == 52
