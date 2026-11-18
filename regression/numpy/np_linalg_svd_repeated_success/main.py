import numpy as np

b = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
u, s, vh = np.linalg.svd(b)
assert abs(s[0] - 1.4142135623730951) < 1e-9 and abs(s[1] - s[0]) < 1e-9
assert abs(s[2] - 1.0) < 1e-9
assert abs(u[0][2] * s[2] * vh[2][2]) < 1e-9
assert abs(abs(vh[2][2]) - 1.0) < 1e-9
