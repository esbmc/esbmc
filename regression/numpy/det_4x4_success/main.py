import numpy as np
m = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
assert np.linalg.det(m) == 1.0
r = np.linalg.det(np.array([[2, 1, 0, 0], [1, 3, 1, 0], [0, 1, 4, 1], [0, 0, 1, 5]]))
assert abs(r - 85.0) < 1e-6
