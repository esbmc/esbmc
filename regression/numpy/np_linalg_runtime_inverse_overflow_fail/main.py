import numpy as np

x = nondet_float()
if x == 7e153:
    b = np.linalg.inv(np.array([[x, 0.0], [0.0, x]]))
    assert b[0][0] > 1.0
