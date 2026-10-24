import numpy as np

a = np.array([1, 3, 5])
x = nondet_int()
__ESBMC_assume(x == 4)
idx = np.searchsorted(a, x)
assert idx == 1
