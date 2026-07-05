import numpy as np

a = np.array([4611686018427387903, 4611686018427387903])
b = np.array([3, 3])
result = np.dot(a, b)
assert result > 0
