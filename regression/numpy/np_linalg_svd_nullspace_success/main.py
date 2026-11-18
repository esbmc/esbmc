import numpy as np

a = np.array([[1.0, 2.0, 3.0]])
u, s, vh = np.linalg.svd(a)
assert abs(u[0][0] * s[0] * vh[0][1] - 2.0) < 1e-9
assert abs(a[0][0] * vh[1][0] + a[0][1] * vh[1][1] + a[0][2] * vh[1][2]) < 1e-9
assert abs(a[0][0] * vh[2][0] + a[0][1] * vh[2][1] + a[0][2] * vh[2][2]) < 1e-9
assert a[0][2] == 3.0

b = np.array([[1.0], [2.0], [2.0]])
p, q, r = np.linalg.svd(b)
assert abs(q[0] - 3.0) < 1e-9
assert abs(b[0][0] * p[0][2] + b[1][0] * p[1][2] + b[2][0] * p[2][2]) < 1e-9

c = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
x, y, z = np.linalg.svd(c, full_matrices=False)
assert y[1] >= 0.0 and y[1] < 1e-9
assert abs(c[1][0] * z[1][0] + c[1][1] * z[1][1] + c[1][2] * z[1][2]) < 1e-9
assert abs(x[0][0] * x[0][1] + x[1][0] * x[1][1]) < 1e-9
