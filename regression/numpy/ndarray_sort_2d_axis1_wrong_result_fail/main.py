import numpy as np

a = np.array([[3, 1], [4, 2]])
a.sort(axis=1)  # In-place sort by row

# Wrong assertion to verify the test detects incorrect values
assert a[0, 0] == 3  # Should be 1
