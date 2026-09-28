import numpy as np

a = np.array([1, 2, 3, 4])
r = np.reshape(a, (2, 2))

r[1][0] = 7

assert a[2] == 7
assert r[1][0] == 7
