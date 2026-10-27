import numpy as np

x = nondet_float()
if x >= 0.5 and x <= 1.0:
    a = np.asin(x)
    b = np.acos(x)
    c = np.atan(x)
    assert a > 0.52 and b < 1.05 and c > 0.46
