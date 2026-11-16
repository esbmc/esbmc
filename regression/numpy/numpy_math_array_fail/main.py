import numpy as np

x = nondet_float()
__ESBMC_assume(x >= 0.5 and x <= 2.0)
a = np.array([1.0, x])
r = np.expm1(a)
assert r[1] < 6.0
