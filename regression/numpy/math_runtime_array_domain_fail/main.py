import numpy as np

x = nondet_float()
if x < 0.0:
    a = np.array([1.0, x])
    r = np.log(a)
    assert r[0] == 0.0 and r[1] == r[1], "domain"
