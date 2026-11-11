import numpy as np

x = nondet_float()
if x >= -3.0 and x <= 3.0:
    v = np.array([x, -4.0])
    n2 = np.linalg.norm(v)
    assert n2 >= 4.0 - 1e-9 and n2 <= 5.0 + 1e-9
    assert np.linalg.norm(v, np.inf) == 4.0
    assert np.linalg.norm(v, 1) >= 4.0 - 1e-9 and np.linalg.norm(v, 1) <= 7.0 + 1e-9
