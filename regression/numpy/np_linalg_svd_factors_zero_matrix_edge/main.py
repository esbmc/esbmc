import numpy as np

u, s, vh = np.linalg.svd(np.array([[0.0, 0.0], [0.0, 0.0]]))
assert s[0] == 0.0 and s[1] == 0.0
assert abs(u[0][0]) == 1.0 and u[1][0] == 0.0 and vh[0][1] == 0.0
assert abs(vh[1][1]) == 1.0
