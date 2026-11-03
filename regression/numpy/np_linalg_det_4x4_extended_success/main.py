import numpy as np

# Past 4x4: the 5x5 tridiagonal matrix of 2s and 1s has determinant 6.
t = np.array([[2.0, 1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 1.0, 0.0, 0.0],
              [0.0, 1.0, 2.0, 1.0, 0.0], [0.0, 0.0, 1.0, 2.0, 1.0],
              [0.0, 0.0, 0.0, 1.0, 2.0]])
d = np.linalg.det(t)
assert abs(d - 6.0) < 1e-9
