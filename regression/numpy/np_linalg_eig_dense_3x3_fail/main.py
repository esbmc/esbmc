import numpy as np

a = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
w, v = np.linalg.eig(a)
assert abs(w[0] * w[1] * w[2] - 19.0) < 1e-9
