import numpy as np

a = np.array([1, 2, 3])
v = np.broadcast_to(a, (2, 3))

a[1] = 9

assert v[0][1] == 9
assert v[1][1] == 9
assert v[1][2] == 3
