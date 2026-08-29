import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
d = np.diagonal(a)
np.fill_diagonal(a, 0)

assert a[1][1] == 0
assert d[1] == 0
