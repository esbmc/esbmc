import numpy as np

a = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [4.0, 0.0, 1.0]])
u, s, vh = np.linalg.svd(a)
assert abs(u[0][0] * s[0] * vh[0][0] + u[0][1] * s[1] * vh[1][0] + u[0][2] * s[2] * vh[2][0] - 1.0) < 1e-9
assert abs(u[1][0] * s[0] * vh[0][2] + u[1][1] * s[1] * vh[1][2] + u[1][2] * s[2] * vh[2][2] - 3.0) < 1e-9
assert abs(u[0][0] * u[0][1] + u[1][0] * u[1][1] + u[2][0] * u[2][1]) < 1e-9
assert abs(vh[0][0] * vh[0][0] + vh[0][1] * vh[0][1] + vh[0][2] * vh[0][2] - 1.0) < 1e-9
