import numpy as np

n = nondet_int()
__ESBMC_assume(n >= 1 and n <= 3)
a = np.ones(n)

assert a[0] == 1.0
