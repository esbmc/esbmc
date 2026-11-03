import numpy as np

x = nondet_float()
if x <= 0.0:
    y = np.log2(x)
    assert y > -1e308, "domain"
