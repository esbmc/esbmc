import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
f = v.flatten()
f[0] = 6
assert a[1] == 6
