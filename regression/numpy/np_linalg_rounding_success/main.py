import numpy as np

m = np.array([[1.0, 2.0], [3.0, 4.0]])
d = np.linalg.det(m)
assert abs(d + 2.0) < 1e-9
assert d != -1.0
r = np.linalg.inv(m)
assert abs(r[1][0] - 1.5) < 1e-9
s = np.linalg.solve(m, np.array([5.0, 6.0]))
assert abs(s[1] - 4.5) < 1e-9

# Factors that are exact, with power-of-two pivots, give NumPy's own value.
assert np.linalg.det(np.array([[-2.0, 1.0], [4.0, -3.0]])) == 2.0
assert np.linalg.inv(np.array([[2.0, 0.0], [0.0, 4.0]]))[1][1] == 0.25
assert np.linalg.solve(np.array([[4.0, 0.0], [0.0, 2.0]]), np.array([1.0, 1.0]))[0] == 0.25
