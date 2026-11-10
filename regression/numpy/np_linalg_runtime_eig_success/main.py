import numpy as np

x = nondet_float()
if x >= 0.5 and x <= 1.0:
    w, v = np.linalg.eig(np.array([[2.0, x], [x, 3.0]]))
    assert abs(2.0 * v[0][0] + x * v[1][0] - w[0] * v[0][0]) < 1e-9
    assert abs(x * v[0][0] + 3.0 * v[1][0] - w[0] * v[1][0]) < 1e-9
    assert abs(2.0 * v[0][1] + x * v[1][1] - w[1] * v[0][1]) < 1e-9
    assert abs(x * v[0][1] + 3.0 * v[1][1] - w[1] * v[1][1]) < 1e-9
    assert abs(v[0][0] * v[0][0] + v[1][0] * v[1][0] - 1.0) < 1e-9
    assert abs(v[0][1] * v[0][1] + v[1][1] * v[1][1] - 1.0) < 1e-9
