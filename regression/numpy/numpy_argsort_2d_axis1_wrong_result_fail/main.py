import numpy as np

a = np.array([[3, 1], [4, 2]])
result = np.argsort(a, axis=1)

# Wrong assertion to verify the test detects incorrect values
assert result[0, 0] == 0  # Should be 1
