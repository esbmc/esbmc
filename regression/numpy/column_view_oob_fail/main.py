import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
col = a[:, 3]

assert col[0] == 1
