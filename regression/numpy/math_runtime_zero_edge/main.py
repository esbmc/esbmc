import numpy as np

x = nondet_float()
if x == 0.0:
    assert np.exp(x) == 1.0
    assert np.cos(x) == 1.0
    assert np.sin(x) == 0.0
    assert np.log(x + 1.0) == 0.0
