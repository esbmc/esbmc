import numpy as np

x = nondet_float()
if x >= 0.5 and x <= 2.0:
    y = np.exp(x)
    assert y > 1.64 and y < 7.39
    s = np.sin(x)
    assert s > 0.47 and s <= 1.0
