import numpy as np

x = nondet_int()
__ESBMC_assume(x > 1 and x < 5)

a = np.array([1, x, 5])
idx = np.searchsorted(a, x)
assert idx == 1
