import numpy as np

n = nondet_int()
__ESBMC_assume(n == 0)
a = np.zeros(n)

assert len(a) == 0
