import numpy as np

a = np.array([1.0, 3.0, 5.0])
x = nondet_float()
__ESBMC_assume(x != x)
idx = np.searchsorted(a, x)
assert idx == 3
