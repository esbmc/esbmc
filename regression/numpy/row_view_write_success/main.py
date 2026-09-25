import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
row[1] = 99

assert a[0][1] == 99
assert row[1] == 99
assert a[1][1] == 5
