import numpy as np

x = nondet_float()
if x >= 1.0 and x <= 2.0:
    b = np.linalg.inv(np.array([[x, 1.0], [1.0, 2.0]]))
    assert abs(x * b[0][0] + b[1][0] - 1.0) < 1e-9
    assert abs(x * b[0][1] + b[1][1]) < 1e-9
    assert abs(b[0][0] + 2.0 * b[1][0]) < 1e-9
    assert abs(b[0][1] + 2.0 * b[1][1] - 1.0) < 1e-9
