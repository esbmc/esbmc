import numpy as np

a = np.array([[3, 1], [2, 4]])
result = np.argsort(a, axis=0)

# Indices for column sort: [[1, 0], [0, 1]]
assert result[0, 0] == 1  # 2 < 3
assert result[0, 1] == 0  # 1 < 4
