import numpy as np

x = nondet_float()
if x >= 0.0 and x <= 1.0:
    u, s, vh = np.linalg.svd(np.array([[x, 0.0]]))
    assert s[0] > 2.0
