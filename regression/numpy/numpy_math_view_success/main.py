import numpy as np

x = nondet_float()
__ESBMC_assume(x >= 1.0 and x <= 2.0)
a = np.array([x, 0.0, 2.0, 0.0])
v = a[::2]
r = np.arccosh(v)
assert r[0] >= 0.0 and r[0] < 1.32
assert r[1] > 1.31 and r[1] < 1.32
t = np.arctanh(a[1:3])
assert t[0] == 0.0
