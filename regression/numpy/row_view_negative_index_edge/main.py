import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[-1]
row[0] = 42

assert a[1][0] == 42
assert row[2] == 6
assert a[0][0] == 1
