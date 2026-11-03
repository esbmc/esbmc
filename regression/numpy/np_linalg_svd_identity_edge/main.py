import numpy as np

a = np.array([[1.0, 0.0], [0.0, 1.0]])
sigma = np.linalg.svd(a, compute_uv=False)

assert abs(sigma[0] - 1.0) < 1e-9
assert abs(sigma[1] - 1.0) < 1e-9
