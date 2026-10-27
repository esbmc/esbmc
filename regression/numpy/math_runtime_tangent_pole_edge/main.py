import numpy as np

x = nondet_float()
if x > 1.5707963267948966 and x <= 2.0:
    assert np.tan(x) < -2.0
if x >= 1.0 and x <= 1.5707963267948966:
    assert np.tan(x) > 1.5
