import numpy as np
m = np.array([[2, 1], [1, 2]])
x = np.linalg.det(m)
assert abs(x - 3) < 1e-9
