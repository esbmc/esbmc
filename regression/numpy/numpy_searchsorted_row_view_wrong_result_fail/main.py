import numpy as np

a = np.array([[1, 3, 5], [2, 4, 6]])
row = a[0]  # [1, 3, 5]
idx = np.searchsorted(row, 4)

# Wrong assertion to verify the test detects incorrect values
assert idx == 1  # Should be 2 (4 would go at index 2)
