import numpy as np

n = nondet_int()
__ESBMC_assume(n >= 1 and n <= 5)
a = np.zeros(n)
mask = np.array([True, False])
b = a[mask]
