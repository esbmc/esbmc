import numpy as np

a = np.array([4611686018427387903, 1])
b = np.array([2, 1])
result = np.dot(a, b)
assert result == 9223372036854775807
