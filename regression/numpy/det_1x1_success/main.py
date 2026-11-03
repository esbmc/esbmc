import numpy as np
m = np.array([[5]])
x = np.linalg.det(m)
assert abs(x - 5.0) < 1e-9
assert np.linalg.det(np.array([[4.0]])) == 4.0
