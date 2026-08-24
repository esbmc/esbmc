import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
d = np.diagonal(a)
a[1][1] = 99

assert d[1] == 99
assert d[0] == 1
assert d[2] == 9
