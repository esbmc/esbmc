import numpy as np

a = np.array([1, 3, 5, 7])
x = nondet_int()
__ESBMC_assume(x == 4)
idx = np.searchsorted(a, [2, x])
assert idx[0] == 1
assert idx[1] == 0
