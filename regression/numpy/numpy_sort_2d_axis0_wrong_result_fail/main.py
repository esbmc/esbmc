import numpy as np

a = np.array([[3, 1], [2, 4]])
result = np.sort(a, axis=0)

# Wrong assertion to verify the test detects incorrect values
assert result[0, 0] == 3  # Should be 2
