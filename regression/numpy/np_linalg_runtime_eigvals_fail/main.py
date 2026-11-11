import numpy as np

x = nondet_float()
if x >= 0.0 and x <= 1.0:
    w = np.linalg.eigvals(np.array([[2.0, x], [x, 3.0]]))
    assert w[0].real * w[1].real > 6.0
