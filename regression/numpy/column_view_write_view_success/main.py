import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
col = a[:, 1]
col[1] = 99

assert a[1][1] == 99
assert col[1] == 99
assert a[0][1] == 2
assert a[2][1] == 8
