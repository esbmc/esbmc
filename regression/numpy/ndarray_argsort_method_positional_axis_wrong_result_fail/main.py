import numpy as np

a = np.array([3, 1, 2])
idx = a.argsort(0)

# Wrong assertion to verify the test detects incorrect values
assert idx[0] == 0  # Should be 1
