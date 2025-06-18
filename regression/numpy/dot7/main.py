import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.array([[1, 0], [0, 1]])
result1 = np.dot(a, b)
result2 = np.dot(result1, c)
# result1 should be [[19, 22], [43, 50]]
assert result2[0][0] == 19
assert result2[0][1] == 22
assert result2[1][0] == 43
assert result2[1][1] == 50
