import numpy as np

x = nondet_float()
y = nondet_float()
if x >= -4.0 and x <= 4.0 and y >= 1.0 and y <= 3.0:
    s = np.linalg.solve(np.array([[3.0, y], [1.0, 4.0]]), np.array([x, 1.0]))
    assert abs(3.0 * s[0] + y * s[1] - x) < 1e-9
    assert abs(s[0] + 4.0 * s[1] - 1.0) < 1e-9
