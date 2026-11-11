import numpy as np

w = np.linalg.eigvals(np.array([[4.0, 1.0, 2.0], [0.5, 3.0, 1.0], [1.0, 0.0, 1.0]]))
assert abs(w[0].real + w[1].real + w[2].real - 8.0) < 1e-9
s = np.linalg.svd(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [4.0, 0.0, 1.0]]), compute_uv=False)
assert abs(s[0] * s[1] * s[2] - 25.0) < 1e-9
x = np.linalg.solve(np.array([[2.0, 1.0], [1.0, 3.0]]), np.array([3.0, 5.0]))
assert abs(x[0] - 0.8) < 1e-9 and abs(x[1] - 1.4) < 1e-9
