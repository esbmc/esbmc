import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
c = v.copy()
b = c
b[0] = 50
assert c[0] == 50
assert v[0] == 1
assert sum(c) == 55
assert sum(v) == 6
