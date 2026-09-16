import numpy as np

a = np.array([[3, 1, 2], [6, 4, 5]])
row = a[0]  # [3, 1, 2]
result = np.sort(row)

assert result[0] == 1
assert result[2] == 3
