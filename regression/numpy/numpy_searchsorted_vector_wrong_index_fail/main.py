import numpy as np

a = np.array([1, 3, 5, 7])
idx = np.searchsorted(a, [2, 6])

# Wrong assertion to verify the test detects incorrect values
assert idx[0] == 0
