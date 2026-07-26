import numpy as np

x = np.linalg.solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))

assert x[0] == 1.0
