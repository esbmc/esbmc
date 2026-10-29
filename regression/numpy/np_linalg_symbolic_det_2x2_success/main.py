import numpy as np

x = nondet_float()
if x >= 1.0 and x <= 10.0:
    d = np.linalg.det(np.array([[x, 1.0], [2.0, 3.0]]))
    assert d > 3.0 * x - 2.0 - 1e-6 and d < 3.0 * x - 2.0 + 1e-6
    assert d >= 1.0 - 1e-6
