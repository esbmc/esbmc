import numpy as np

a = np.array([[1, 3], [2, 4]])
col = a[:, 1]  # [3, 4]

# Wrong assertion to verify the test detects incorrect values
assert np.median(col) == 4  # Should be 3.5
