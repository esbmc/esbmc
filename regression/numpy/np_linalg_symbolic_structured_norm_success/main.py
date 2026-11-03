import numpy as np

x = nondet_float()
if x >= -1.0 and x <= 1.0:
    f = np.linalg.norm(np.array([[x, 0.0], [0.0, 2.0]]))
    assert f >= 2.0 - 1e-9 and f <= 2.24
