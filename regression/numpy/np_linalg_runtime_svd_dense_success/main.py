import numpy as np

x = nondet_float()
if x >= 0.0 and x <= 1.0:
    a = np.array([[x, 1.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
    s = np.linalg.svd(a, compute_uv=False)
    assert s[0] >= s[1] and s[1] >= s[2] and s[2] >= 0.0
    assert abs(s[0] * s[0] + s[1] * s[1] + s[2] * s[2] - (x * x + 32.0)) < 1e-6
    b = np.array([[x, 1.0, 2.0], [3.0, 4.0, 5.0]])
    u, t, vh = np.linalg.svd(b, full_matrices=False)
    assert len(u) == 2 and len(u[0]) == 2 and len(vh) == 2 and len(vh[0]) == 3
    assert abs(u[1][0] * t[0] * vh[0][2] + u[1][1] * t[1] * vh[1][2] - 5.0) < 1e-6
    assert abs(vh[0][0] * vh[1][0] + vh[0][1] * vh[1][1] + vh[0][2] * vh[1][2]) < 1e-6
