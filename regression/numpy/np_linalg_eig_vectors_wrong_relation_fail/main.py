import numpy as np

a = np.array([[4.0, 1.0], [2.0, 3.0]])
w, v = np.linalg.eig(a)
r = a[0][0] * v[0][0] + a[0][1] * v[1][0] - w[1] * v[0][0]
assert abs(r) < 1e-9, "relation"
