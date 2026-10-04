import numpy as np

a = np.array([[1, 2, 3]])
s = np.squeeze(a)
assert s[2] == 3
s[0] = 10
assert a[0][0] == 10
e = np.expand_dims(s, 0)
assert e.shape == (1, 3)
assert e[0][1] == 2
b = np.broadcast_to(s, (2, 3))
assert b[1][2] == 2
a[0][2] = 33
assert b[0][2] == 33
