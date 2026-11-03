import numpy as np

x = nondet_float()
if x >= 0.5 and x <= 2.0:
    a = np.array([[x, 1.0], [2.0, -x]])
    b = np.exp(a)
    assert b[0][0] > 1.64 and b[0][1] > 2.71 and b[1][1] < 0.61
