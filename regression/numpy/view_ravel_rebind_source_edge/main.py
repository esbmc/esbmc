import numpy as np

a = np.array([[1, 2], [3, 4]])
r = np.ravel(a)
a = np.array([[9, 9], [9, 9]])

assert r[0] == 1
assert r[3] == 4
assert a[0][0] == 9
