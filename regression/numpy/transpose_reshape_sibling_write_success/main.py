import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)
r = np.reshape(a, (4,))

t[1][0] = 9

assert a[0][1] == 9
assert r[1] == 9

