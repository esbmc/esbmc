import numpy as np

a = np.array([[1, 2, 3]])
v = np.squeeze(a, 0)

v[1] = 9

assert a[0][0] == 1
assert a[0][1] == 9
assert v[1] == 9
