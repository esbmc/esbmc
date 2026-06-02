import numpy as np

m = np.array([[1, 2], [3, 4]])
col = m[:, 0]
assert col[0] == 1
