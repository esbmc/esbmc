import numpy as np

n = nondet_int()
__ESBMC_assume(n >= 1 and n <= 3)
a = np.array([1.0, 8.0, 27.0])
r = np.cbrt(a[:n])
assert r[0] > 1.0
