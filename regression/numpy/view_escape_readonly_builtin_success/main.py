import numpy as np

a = np.array([[1, 2], [3, 4]])
row = a[0]
n = len(row)
shape = row.shape

assert n == 2
assert shape[0] == 2
