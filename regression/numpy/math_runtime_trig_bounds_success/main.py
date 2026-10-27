import numpy as np

x = nondet_float()
if x > 1.0:
    a = np.arccos(x)
    assert a != a
if x >= 1e10 and x < float("inf"):
    c = np.cos(x)
    assert c >= -1.0 and c <= 1.0
