import numpy as np

a = np.array([[3, 1], [4, 2]])
result = np.argsort(a, axis=1)

# Indices for row sort: [[1, 0], [1, 0]]
assert result[0, 0] == 1  # 1 < 3
assert result[1, 0] == 1  # 2 < 4
