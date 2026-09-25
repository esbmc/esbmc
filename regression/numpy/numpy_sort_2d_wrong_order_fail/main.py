import numpy as np

a = np.array([[3, 1], [4, 2]])
result = np.sort(a, axis=1)

# Wrong assertion to verify test detects incorrect values
assert result[0, 0] == 3  # Should be 1
