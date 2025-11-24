import numpy as np
singular = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
assert abs(np.linalg.det(singular)) < 1e-15
