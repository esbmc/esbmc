import numpy as np

x = nondet_float()
if x >= 1.0 and x <= 2.0:
    b = np.linalg.inv(np.array([[x, 1.0], [1.0, 2.0]]))
    assert b[0][0] > 1.0
