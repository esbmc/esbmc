import numpy as np
r = np.linalg.det(np.array([[2, 1, 0, 0], [1, 3, 1, 0], [0, 1, 4, 1], [0, 0, 1, 5]]))
assert r == 85.0
