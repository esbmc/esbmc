import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
t = a.T
r = np.ravel(t)
r[0] = 99
assert a[0][0] == 99
