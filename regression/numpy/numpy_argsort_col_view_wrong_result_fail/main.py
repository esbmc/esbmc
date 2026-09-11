import numpy as np

a = np.array([[3, 1], [2, 4]])
col = a[:, 0]  # [3, 2]
idx = np.argsort(col)

# Wrong assertion to verify the test detects incorrect values
assert idx[0] == 0  # Should be 1 (2 < 3)
