import numpy as np

b = np.array([[3.0, 1.0], [1.0, 4.0], [2.0, 0.0]])
u, s, vh = np.linalg.svd(b)
assert abs(u[2][0] * s[0] * vh[0][0] + u[2][1] * s[1] * vh[1][0] - 1.0) < 1e-9
