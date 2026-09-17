import numpy as np

a = np.array([[1], [2], [3]])
row = a[1]
row[0] = 20

assert a[1][0] == 20
assert row[0] == 20
assert a[0][0] == 1
