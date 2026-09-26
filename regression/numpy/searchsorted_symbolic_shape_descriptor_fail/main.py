import numpy as np

n = nondet_int()
__ESBMC_assume(n >= 1 and n <= 4)

a = np.zeros(n)
idx = np.searchsorted(a, 0)
