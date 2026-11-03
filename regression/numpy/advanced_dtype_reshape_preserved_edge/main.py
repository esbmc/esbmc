import numpy as np

a = np.arange(6, dtype=float)
r = a.reshape((2, 3))
assert r[1][2] == 5.0
a[5] = 1.5
assert r[1][2] == 1.5
