import numpy as np

y = np.array([3.0, 3.0, -3.0, -3.0])
x = np.array([4.0, -4.0, -4.0, 4.0])
t = np.arctan2(y, x)
assert abs(t[0] - 0.6435011087932844) < 1e-12
assert abs(t[1] - 2.498091544796509) < 1e-12
assert abs(t[2] + 2.498091544796509) < 1e-12
assert abs(t[3] + 0.6435011087932844) < 1e-12
v = np.arctan2(y[1:3], x[1:3])
assert abs(v[1] + 2.498091544796509) < 1e-12
