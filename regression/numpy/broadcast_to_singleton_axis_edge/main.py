import numpy as np

a = np.array([[1], [2]])
v = np.broadcast_to(a, (2, 3))

a[1][0] = 8

assert v[0][0] == 1
assert v[0][2] == 1
assert v[1][0] == 8
assert v[1][2] == 8
