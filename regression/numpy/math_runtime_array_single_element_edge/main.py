import numpy as np

x = nondet_float()
if x >= 1.0 and x <= 2.0:
    a = np.array([[x]])
    s = np.sin(a)
    assert s[0][0] > 0.84
