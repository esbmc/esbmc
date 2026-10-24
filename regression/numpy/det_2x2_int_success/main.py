import numpy as np
m = np.array([[1, 2], [3, 4]])
x = np.linalg.det(m)
assert abs(x + 2) < 1e-9
