import numpy as np

a = np.array([[1, 2, 3]])
v = np.squeeze(a, -2)

v[2] = 5

assert a[0][2] == 5
assert v[0] == 1
