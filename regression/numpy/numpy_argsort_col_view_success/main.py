import numpy as np

a = np.array([[3, 1], [2, 4]])
col = a[:, 0]  # [3, 2]
idx = np.argsort(col)

assert idx[0] == 1  # 2 < 3
assert idx[1] == 0
