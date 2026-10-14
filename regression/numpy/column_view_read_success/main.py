import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
col = a[:, 1]

assert col[0] == 2
assert col[1] == 5
assert col[2] == 8
