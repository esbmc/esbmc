import numpy as np

a = np.array([[4.0, 7.0], [2.0, 6.0]])
b = np.linalg.inv(a)
assert abs(a[0][0] * b[0][0] + a[0][1] * b[1][0] - 1.0) < 1e-9
assert abs(a[1][0] * b[0][1] + a[1][1] * b[1][1] - 1.0) < 1e-9
assert abs(a[0][0] * b[0][1] + a[0][1] * b[1][1]) < 1e-9
