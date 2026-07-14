import numpy as np

a = np.array([2])
b = np.array([[3, 4]])
result = np.dot(a, b)

assert result[0] == 6
assert result[1] == 8
