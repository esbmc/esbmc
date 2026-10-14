import numpy as np

a = np.array([[3, 1], [2, 4]])
a.sort(axis=0)  # In-place sort by column

# Wrong assertion to verify the test detects incorrect values
assert a[0, 0] == 3  # Should be 2
