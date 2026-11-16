import numpy as np

x = nondet_float()
if x >= -1e100 and x <= 1e100:
    d = np.linalg.det(np.array([[x, 0.0], [0.0, 0.0]]))
    assert d == 0.0
