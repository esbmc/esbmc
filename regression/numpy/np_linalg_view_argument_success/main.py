import numpy as np

a = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
d = np.linalg.det(a[0:2, 0:2])
assert abs(d - 5.0) < 1e-9
n = np.linalg.norm(a[1])
assert abs(n - 3.1622776601683795) < 1e-9
