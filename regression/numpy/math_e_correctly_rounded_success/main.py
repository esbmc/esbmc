import numpy as np

assert np.exp(1.0) >= np.e
x = nondet_float()
if x >= 1.0 and x <= 2.0:
    assert np.exp(x) >= np.e
