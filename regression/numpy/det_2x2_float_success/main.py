import numpy as np
m = np.array([[1.5, 2.0], [3.0, 4.0]])
x = np.linalg.det(m)
assert abs(x) < 1e-9
