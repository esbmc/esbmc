import numpy as np

x = nondet_float()
if x >= 0.0 and x <= 1.0:
    a = np.array([[2.0, 1.0, x], [0.0, 3.0, 1.0], [0.0, 0.0, 5.0]])
    w = np.linalg.eigvals(a)
    assert abs((w[0] + w[1] + w[2]).real - 9.0) < 1e-6
