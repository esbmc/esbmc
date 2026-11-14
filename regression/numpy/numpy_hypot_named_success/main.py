import numpy as np

a = np.array([3.0, 5.0])
b = np.array([4.0, 12.0])
h = np.hypot(a, b)
assert abs(h[0] - 5.0) < 1e-12
assert abs(h[1] - 13.0) < 1e-12
g = np.hypot(a[::-1], b[::-1])
assert abs(g[0] - 13.0) < 1e-12
