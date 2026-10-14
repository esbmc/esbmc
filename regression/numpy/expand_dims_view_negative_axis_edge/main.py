import numpy as np

a = np.array([1, 2, 3])
v = np.expand_dims(a, -1)

v[1][0] = 4

assert a[1] == 4
assert v[2][0] == 3
