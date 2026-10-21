import numpy as np

a = np.array([1, 2])
f = np.array([0.5, 1.5])
s = a + a
assert s[0] == 2 and s[1] == 4
m = a * f
assert m[0] == 0.5 and m[1] == 3.0
d = a - a
assert d[1] == 0
t = 2 * a
assert t[1] == 4
r = a / a
assert r[1] == 1.0
