import numpy as np

a = np.array([[1.0, 2.0, 1.0], [-2.0, 1.0, 0.0], [1.0, 0.0, 3.0]])
w, v = np.linalg.eig(a)
r = a[0][0] * v[0][1] + a[0][1] * v[1][1] + a[0][2] * v[2][1] - w[0] * v[0][1]
s = a[1][0] * v[0][1] + a[1][1] * v[1][1] + a[1][2] * v[2][1] - w[0] * v[1][1]
assert abs(r.real) < 1e-6 and abs(r.imag) < 1e-6 and abs(s.real) < 1e-6 and abs(s.imag) < 1e-6
