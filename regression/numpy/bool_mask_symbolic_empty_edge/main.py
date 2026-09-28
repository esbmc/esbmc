import numpy as np

a = np.array([7])
n = nondet_bool()
__ESBMC_assume(not n)
mask = np.array([n])
b = a[mask]

assert len(b) == 0
