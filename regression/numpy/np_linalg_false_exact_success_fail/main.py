import numpy as np

x = np.linalg.solve(np.array([[3.0, 1.0], [1.0, 2.0]]), np.array([9.0, 8.0]))
assert x[0] == 2.0, "exact"
