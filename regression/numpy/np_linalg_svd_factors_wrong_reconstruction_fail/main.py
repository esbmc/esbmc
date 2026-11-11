import numpy as np

a = np.array([[3.0, 0.0], [4.0, 5.0]])
u, s, vh = np.linalg.svd(a)
r = u[0][0] * s[1] * vh[0][0] + u[0][1] * s[0] * vh[1][0]
assert abs(r - 3.0) < 1e-9, "reconstruction"
