import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])
row = a[2]
row[1] = 60

assert a[2][1] == 60
assert row[0] == 5
assert a[1][1] == 4
