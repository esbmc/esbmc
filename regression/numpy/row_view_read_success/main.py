import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]

assert row[0] == 1
assert row[1] == 2
assert row[2] == 3
assert a[0][1] == 2
