import numpy as np

a = np.array([[4.0, 1.0], [2.0, 3.0]])
w, v = np.linalg.eig(a)
assert len(w) == 2 and len(v) == 2 and len(v[0]) == 2
r0 = a[0][0] * v[0][0] + a[0][1] * v[1][0] - w[0] * v[0][0]
r1 = a[1][0] * v[0][0] + a[1][1] * v[1][0] - w[0] * v[1][0]
s0 = a[0][0] * v[0][1] + a[0][1] * v[1][1] - w[1] * v[0][1]
assert abs(r0) < 1e-9 and abs(r1) < 1e-9 and abs(s0) < 1e-9
assert abs(v[0][1] * v[0][1] + v[1][1] * v[1][1] - 1.0) < 1e-9
