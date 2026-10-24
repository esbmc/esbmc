import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
c = np.copy(v)
v[1] = 0
assert c[1] == 0
