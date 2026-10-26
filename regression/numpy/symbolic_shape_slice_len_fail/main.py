import numpy as np

n = nondet_int()
__ESBMC_assume(n >= 1 and n <= 3)
b = np.ones(n)
k = nondet_int()
__ESBMC_assume(k >= 0 and k <= 3)
r = b[0:k]
assert len(r) == k
