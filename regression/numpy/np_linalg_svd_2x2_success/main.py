import numpy as np

a = np.array([[3.0, 0.0], [0.0, 2.0]])
sigma = np.linalg.svd(a, compute_uv=False)

assert abs(sigma[0] - 3.0) < 1e-9
assert abs(sigma[1] - 2.0) < 1e-9
