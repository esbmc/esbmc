import numpy as np

x = np.array([1.0, 8.0, -27.0])
o = np.zeros(3)
np.cbrt(x, where=[True, False, True], out=o)
assert o[0] > 0.99 and o[0] < 1.01 and o[1] == 0.0 and o[2] < -2.99 and o[2] > -3.01
p = np.zeros(3)
np.cbrt(x, where=(False, True, 0), out=p)
assert p[0] == 0.0 and p[1] > 1.99 and p[1] < 2.01 and p[2] == 0.0
q = np.zeros(3)
np.cbrt(x, where=~(x > 0.0) | (x > 5.0), out=q)
assert q[0] == 0.0 and q[1] > 1.99 and q[1] < 2.01 and q[2] < -2.99 and q[2] > -3.01
r = np.zeros(3)
np.cbrt(x, where=(x > 0.0) & (x < 5.0), out=r)
assert r[0] > 0.99 and r[0] < 1.01 and r[1] == 0.0 and r[2] == 0.0
s = np.full(3, 4.0)
np.cbrt(x, where=None, out=s)
assert s[0] == 4.0 and s[2] == 4.0
