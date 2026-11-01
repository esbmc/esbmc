import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.swapaxes(a, 0, 1)

t[1][0] = 7

assert a[0][1] == 7
assert t[1][0] == 7

p = np.array([[1, 2], [3, 4]])
s = np.swapaxes(p, 1, 1)

s[0][1] = 6

assert p[0][1] == 6
assert s[0][1] == 6
