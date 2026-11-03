import numpy as np

x = nondet_float()
if x >= -2.0 and x <= -1.0:
    c = np.cosh(x)
    assert c > 1.54 and c < 3.77
    a = np.arccos(x / 2.0)
    assert a >= 2.09 and a <= 3.1415926535897931
