import numpy as np

x = nondet_float()
r = np.sqrt(x)
if x < 0.0:
    assert r != r
if x == 4.0:
    assert r == 2.0
