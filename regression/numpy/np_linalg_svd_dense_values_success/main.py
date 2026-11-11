import numpy as np

s = np.linalg.svd(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [4.0, 0.0, 1.0]]), compute_uv=False)
assert s[0] >= s[1] and s[1] >= s[2] and s[2] >= 0.0
assert abs(s[0] * s[1] * s[2] - 25.0) < 1e-9
assert abs(s[0] * s[0] + s[1] * s[1] + s[2] * s[2] - 32.0) < 1e-9
