import numpy as np

a = np.array([[1, 2, 3]])
v = np.squeeze(a, 0)

a[0][2] = 8

assert v[0] == 1
assert v[2] == 8
