import numpy as np

x = nondet_float()
if x < -1.0:
    a = np.arcsin(x)
    assert a >= -1.5707963267948966, "domain"
