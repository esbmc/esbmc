import numpy as np

a = np.array([[1, 3], [4, 2]])
col = a[:, 0]  # [1, 4]

# Wrong assertion to verify the test detects incorrect values
assert np.argmax(col) == 0  # Should be 1
