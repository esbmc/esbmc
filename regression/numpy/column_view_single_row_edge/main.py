import numpy as np

a = np.array([[5, 6, 7]])
col = a[:, 2]
col[0] = 42

assert a[0][2] == 42
assert len(col) == 1
