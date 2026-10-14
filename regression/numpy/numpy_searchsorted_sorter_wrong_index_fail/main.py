import numpy as np

a = np.array([5, 1, 3])
idx = np.searchsorted(a, 2, sorter=np.argsort(a))

# Wrong assertion to verify the test detects incorrect values
assert idx == 0
