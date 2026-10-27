import numpy as np

a = np.array([1, 2, 3, 4])
v = a[1:3]
c = a[1:3].copy()
for x, y in np.nditer([v, c], op_flags=[["readwrite"], ["readwrite"]]):
    x[...] = x * 10
    y[...] = 0
assert a[1] == 20
assert a[2] == 30
assert a[3] == 4
assert c[0] == 0
assert c[1] == 0
