import numpy as np

a = np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 0.0]])
u, s, vh = np.linalg.svd(a)
assert len(u) == 2 and len(u[0]) == 2 and len(s) == 2
assert len(vh) == 3 and len(vh[0]) == 3
assert s[0] >= s[1] and s[1] >= 0.0
assert abs(s[0] - 4.760137661) < 1e-6 and abs(s[1] - 2.888094443) < 1e-6
assert abs(u[0][0] * s[0] * vh[0][1] + u[0][1] * s[1] * vh[1][1] - 1.0) < 1e-9
assert abs(vh[0][0] * vh[2][0] + vh[0][1] * vh[2][1] + vh[0][2] * vh[2][2]) < 1e-9

b = np.array([[3.0, 1.0], [1.0, 4.0], [2.0, 0.0]])
p, q, r = np.linalg.svd(b, False)
assert len(p) == 3 and len(p[0]) == 2 and len(r) == 2
assert abs(p[2][0] * q[0] * r[0][0] + p[2][1] * q[1] * r[1][0] - 2.0) < 1e-9
assert abs(q[0] - s[0]) < 1e-9 and abs(q[1] - s[1]) < 1e-9

c = np.linalg.svd(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]), compute_uv=False)
assert len(c) == 2 and abs(c[0] * c[0] - 70.0) < 1e-9 and c[1] < 1e-9
