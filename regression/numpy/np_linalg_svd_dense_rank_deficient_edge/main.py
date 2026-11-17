import numpy as np

s = np.linalg.svd(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), compute_uv=False)
assert s[2] >= 0.0 and s[2] < 1e-10
assert abs(s[0] * s[0] + s[1] * s[1] + s[2] * s[2] - 285.0) < 1e-9
