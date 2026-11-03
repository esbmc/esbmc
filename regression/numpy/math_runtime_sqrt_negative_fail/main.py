import numpy as np

x = nondet_float()
if x < 0.0:
    r = np.sqrt(x)
    assert r >= 0.0, "domain"
