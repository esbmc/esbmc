import numpy as np

a = np.array([[3, 1], [4, 2]])
result = np.sort(a, axis=2)  # Out of range for 2-D array

assert result[0, 0] == 1
