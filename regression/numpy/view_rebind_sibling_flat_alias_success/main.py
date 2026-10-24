import numpy as np

a = np.array([1, 2, 3, 4])
f = a.flat
s = a[1:3]
a = np.array([7, 7, 7, 7])
s[0] = 9
assert f[1] == 9
assert f[3] == 4
assert a[1] == 7
