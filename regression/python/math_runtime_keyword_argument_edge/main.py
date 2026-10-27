import numpy as np

x = nondet_float()
if x >= 0.5 and x <= 2.0:
    y = np.exp(x, out=None)
    s = np.sin(x, dtype=float)
    assert y > 1.64 and s > 0.47
