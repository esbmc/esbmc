import numpy as np

a = np.array([[3, 1, 2], [6, 4, 5]])
row = a[0]  # [3, 1, 2]
result = np.sort(row)

# Wrong assertion to verify the test detects incorrect values
assert result[0] == 3  # Should be 1
