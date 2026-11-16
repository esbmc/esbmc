import numpy as np

# Singular for every x, but LU rounds: NumPy gives 8.2e-16 for x = 3.7.
x = nondet_float()
if x >= 1.0 and x <= 10.0:
    d = np.linalg.det(np.array([[x, 2.0 * x], [1.0, 2.0]]))
    assert d > -1e-8 and d < 1e-8
