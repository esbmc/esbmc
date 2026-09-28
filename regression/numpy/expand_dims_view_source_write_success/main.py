import numpy as np

a = np.array([1, 2, 3])
v = np.expand_dims(a, 0)

a[2] = 6

assert v[0][0] == 1
assert v[0][2] == 6
