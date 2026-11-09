import numpy as np

x = nondet_float()
if x >= 1.0 and x <= 2.0:
    assert np.expm1(x) < 6.0, "bound"
