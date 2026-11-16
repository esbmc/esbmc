import numpy as np

x = nondet_float()
if x >= 0.0 and x <= 1.0:
    a = np.array([[x, 1.0, 0.0], [0.0, 2.0, 1.0]])
    s = np.linalg.svd(a, compute_uv=False)
    assert s[0] * s[0] + s[1] * s[1] < 6.0
