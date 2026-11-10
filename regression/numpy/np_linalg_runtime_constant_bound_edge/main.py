import numpy as np

x = nondet_float()
if x == 2.0:
    r = np.linalg.inv(np.array([[x, 1.0], [1.0, 2.0]]))
    c = np.linalg.inv(np.array([[2.0, 1.0], [1.0, 2.0]]))
    assert abs(r[0][0] - c[0][0]) < 1e-9 and abs(r[1][0] - c[1][0]) < 1e-9
    d = np.linalg.det(np.array([[x, 2.0 * x], [1.0, 2.0]]))
    assert abs(d) < 1e-9
