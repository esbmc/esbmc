import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)
r = np.reshape(t, (4,))
a[0][0] = 9

assert r[0] == 1
assert r[1] == 3
assert r[2] == 2
