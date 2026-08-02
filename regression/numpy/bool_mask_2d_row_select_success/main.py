import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
mask = np.array([True, False, True])
b = a[0][mask]

assert b[0] == 1
assert b[1] == 3
