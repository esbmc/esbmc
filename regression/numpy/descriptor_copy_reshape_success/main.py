import numpy as np

a = np.array([1, 2, 3, 4])
r = np.reshape(a, (2, 2))
b = r.copy()

a[1] = 9
b[0][1] = 8

assert a[1] == 9
assert b[0][1] == 8
assert b[1][0] == 3
