import numpy as np

a = np.array([1, 2, 3])
v = np.expand_dims(a, 0)

v[0][1] = 7

assert a[0] == 1
assert a[1] == 7
assert v[0][1] == 7
