import numpy as np

x = nondet_float()
if x == 1e155:
    s = np.linalg.svd(np.array([[x, 0.0], [0.0, x]]), compute_uv=False)
    assert s[0] < 1.0
