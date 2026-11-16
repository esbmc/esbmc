import numpy as np

# sigma1 / sigma2 = 1e12: u2 must not inherit A v2's rounding scaled by it.
u, s, vh = np.linalg.svd(np.array([[1.0, 1000000.0], [0.0, 1.0]]))
assert abs(u[0][1]) > 0.9e-6 and abs(u[0][1]) < 1.1e-6
