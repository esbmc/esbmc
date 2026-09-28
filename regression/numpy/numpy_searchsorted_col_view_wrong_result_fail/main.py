import numpy as np

a = np.array([[1, 3], [2, 4]])
col = a[:, 0]  # [1, 2]
idx = np.searchsorted(col, 1.5)

# Wrong assertion to verify the test detects incorrect values
assert idx == 0  # Should be 1
