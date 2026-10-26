import numpy as np

a = np.array([[1e80, 0.0, 0.0, 0.0], [0.0, 3e-79, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0],
              [0.0, 0.0, 0.0, 3.0]])
d = np.linalg.det(a)
assert d == d or d != d
