import numpy as np

x = nondet_float()
if x >= 1e200 and x <= 1e300:
    d = np.linalg.det(np.array([[x, 0.0], [0.0, x]]))
    assert d < 1e300, "overflow"
