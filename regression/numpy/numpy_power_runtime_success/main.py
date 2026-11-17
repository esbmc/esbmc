import numpy as np

x = nondet_float()
if x >= 1.0 and x <= 2.0:
    a = np.array([x, 2.0])
    r = np.power(a, 2.0)
    assert abs(r[0] - x * x) <= 1e-12 * r[0]
    assert r[1] == 4.0
