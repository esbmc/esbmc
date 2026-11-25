import numpy as np

a = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
u, s, vh = np.linalg.svd(a)
assert abs(s[0] - 14.0) < 1e-9 and s[1] < 1e-9 and s[2] >= 0.0
assert abs(a[0][0] * u[0][1] + a[1][0] * u[1][1] + a[2][0] * u[2][1]) < 1e-9
assert abs(a[2][0] * vh[2][0] + a[2][1] * vh[2][1] + a[2][2] * vh[2][2]) < 1e-9
