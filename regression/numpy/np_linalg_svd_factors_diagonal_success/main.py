import numpy as np

u, s, vh = np.linalg.svd(np.array([[-2.0, 0.0], [0.0, 5.0]]))
assert s[0] == 5.0 and s[1] == 2.0
assert u[0][0] == 0.0 and abs(u[1][0]) == 1.0 and abs(u[0][1]) == 1.0
assert u[1][0] * vh[0][1] == 1.0 and u[0][1] * vh[1][0] == -1.0
