import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 2], [3, 4]])

result = np.dot(a, b)

assert result[0][0] == 1
