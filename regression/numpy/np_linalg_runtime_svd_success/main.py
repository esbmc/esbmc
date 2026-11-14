import numpy as np

x = nondet_float()
if x >= -2.0 and x <= 2.0:
    s = np.linalg.svd(np.array([[x, 1.0], [2.0, 3.0]]), compute_uv=False)
    assert s[0] >= s[1] and s[1] >= 0.0
    assert abs(s[0] * s[1] - abs(3.0 * x - 2.0)) < 1e-9
    assert abs(s[0] * s[0] + s[1] * s[1] - (x * x + 14.0)) < 1e-9
