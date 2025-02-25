import numpy as np

a = np.subtract(4.0, 1.0)
assert a == 3.0

b = np.subtract(1.0, 4.0)
assert b == -3.0

c = np.subtract(5,3)
assert c == 2

d = np.subtract([3,4],[1,3])
assert d[0] == 2
assert d[1] == 1

e = np.array([5,4])
f = np.array([2,2])
g = np.subtract(e, f)
assert g[0] == 3
assert g[1] == 2