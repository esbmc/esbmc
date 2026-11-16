import numpy as np

x = nondet_float()
if x > 700.0:
    assert np.exp(x) < 1e308
