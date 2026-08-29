import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
a[0][2] = 77

assert row[2] == 77
assert a[0][2] == 77
assert a[1][2] == 6
