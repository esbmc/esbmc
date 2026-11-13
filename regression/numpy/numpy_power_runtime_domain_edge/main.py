import numpy as np

x = nondet_float()
if x >= -2.0 and x <= -1.0:
    r = np.power(np.array([x]), 0.5)
    assert np.isnan(r[0])
