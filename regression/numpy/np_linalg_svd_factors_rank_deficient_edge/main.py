import numpy as np

u, s, vh = np.linalg.svd(np.array([[1.0, 2.0], [2.0, 4.0]]))
assert s[1] >= 0.0 and s[1] < 1e-11
r = u[0][0] * s[0] * vh[0][0]
assert abs(r - 1.0) < 1e-9
