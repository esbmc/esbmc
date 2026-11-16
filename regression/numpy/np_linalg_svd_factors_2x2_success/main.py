import numpy as np

a = np.array([[3.0, 0.0], [4.0, 5.0]])
u, s, vh = np.linalg.svd(a)
assert len(u) == 2 and len(s) == 2 and len(vh) == 2
assert s[0] >= s[1]
r = u[0][0] * s[0] * vh[0][0] + u[0][1] * s[1] * vh[1][0]
assert abs(r - 3.0) < 1e-9
assert abs(u[0][0] * u[0][0] + u[1][0] * u[1][0] - 1.0) < 1e-9
