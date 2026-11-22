import numpy as np

x = np.array([0.0, 1.0])
o = np.zeros(2)
r = np.exp(x, o)
assert o[0] == 1.0 and r[0] == 1.0
p = np.zeros(2)
np.sqrt(np.array([4.0, 9.0]), out=(p,))
assert p[0] == 2.0 and p[1] == 3.0
a = np.array([0.0, 1.0])
np.expm1(a, out=a)
assert a[0] == 0.0
assert a[1] > 1.71 and a[1] < 1.72
b = np.zeros(2, dtype=int)
np.isnan(np.array([0.0, np.nan]), out=b)
assert b[0] == 0 and b[1] == 1
c = np.zeros(3)
np.hypot(x, x, c[1:])
assert c[0] == 0.0 and c[1] == 0.0
assert c[2] >= 1.0 and c[2] < 1.5
