import numpy as np

r = np.linalg.solve(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([5.0, 6.0]))
assert r[0] == -4.0
