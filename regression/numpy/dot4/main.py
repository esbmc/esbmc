import numpy as np
identity = np.array([[1, 0], [0, 1]])
matrix = np.array([[5, 7], [3, 2]])
result = np.dot(identity, matrix)
assert result[0][0] == 5
assert result[0][1] == 7
assert result[1][0] == 3
assert result[1][1] == 2
