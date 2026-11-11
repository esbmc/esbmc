import numpy as np

x = nondet_float()
if x > 1.5 and x < 1.6:
    assert np.tan(x) < 1e16, "unbounded"
