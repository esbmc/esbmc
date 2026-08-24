import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
col = a[:, -1]
col[0] = 99

assert a[0][2] == 99
assert col[1] == 6
