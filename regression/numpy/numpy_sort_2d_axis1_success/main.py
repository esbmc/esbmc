import numpy as np

a = np.array([[3, 1], [4, 2]])
result = np.sort(a, axis=1)

# axis=1 sorts each row: [[1, 3], [2, 4]]
assert result[0, 0] == 1
assert result[0, 1] == 3
assert result[1, 0] == 2
assert result[1, 1] == 4
