import numpy as np

a = np.array([[3, 1], [2, 4]])
result = np.sort(a, axis=0)

# axis=0 sorts each column: [[2, 1], [3, 4]]
assert result[0, 0] == 2
assert result[0, 1] == 1
assert result[1, 0] == 3
assert result[1, 1] == 4
