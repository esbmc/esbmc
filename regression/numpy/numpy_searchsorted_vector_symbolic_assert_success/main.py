import numpy as np

a = np.array([1, 3, 5, 7])
x = nondet_int()
__ESBMC_assume(x >= 0)
__ESBMC_assume(x <= 8)
idx = np.searchsorted(a, [2, x])
assert idx[0] == 1
assert idx[1] >= 0
assert idx[1] <= 4
