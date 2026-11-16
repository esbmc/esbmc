import numpy as np

x = nondet_float()
if x == 1e110:
    w = np.linalg.eigvals(np.array([[x, 0.0, 0.0], [0.0, x, 0.0], [0.0, 0.0, x]]))
    assert w[0].real < 1.0
