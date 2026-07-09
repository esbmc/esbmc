import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])
row = a[1]
assert row[0] == 3
assert row[1] == 4
