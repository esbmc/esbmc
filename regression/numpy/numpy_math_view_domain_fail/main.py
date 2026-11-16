import numpy as np

x = nondet_float()
__ESBMC_assume(x >= 0.5 and x <= 2.0)
a = np.array([1.5, x, 3.0])
r = np.arccosh(a[1:])
assert r[0] == r[0]
