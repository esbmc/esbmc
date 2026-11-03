import numpy as np

a = np.array([1, 2])
f = np.array([2.0, 4.0])
b = a - 0.5
assert b[0] == 0.5
c = a / 2
assert c[0] == 0.5 and c[1] == 1.0
d = 1.0 / f
assert d[1] == 0.25
e = 10 - a
assert e[0] == 9 and e[1] == 8
g = f / 0.0
assert g[0] > 1e308
