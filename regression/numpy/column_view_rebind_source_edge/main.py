import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
col = a[:, 1]
a = np.array([[9, 9, 9], [9, 9, 9]])

assert col[0] == 2
assert col[1] == 5
assert a[0][1] == 9
