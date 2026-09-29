import numpy as np

a = np.array([[1, 3], [2, 4]])
col = a[:, 0]  # [1, 2]
idx = np.searchsorted(col, [0, 1.5, 3])
assert idx[0] == 0
assert idx[1] == 1
assert idx[2] == 2
