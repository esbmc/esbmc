import numpy as np

a = np.divide(2.0, 4.0)
assert a == 0.5

b = np.divide(4.0, 2.0)
assert b == 2

c = np.divide([6,8],[2,4])
assert c[0] == 3
assert c[1] == 2

d = np.array([6,4])
e = np.array([2,2])
f = np.divide(d, e)
assert f[0] == 3
assert f[1] == 2
