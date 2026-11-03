import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
xa = a[1:3]
ya = a[1:3]
xb = b[1:3]
yb = b[1:3]
a = np.array([0, 0, 0, 0])
b = np.array([0, 0, 0, 0])
xa[0] = 70
assert ya[0] == 70
assert yb[0] == 6
xb[0] = 50
assert yb[0] == 50
assert ya[0] == 70
assert a[1] == 0
assert b[1] == 0
