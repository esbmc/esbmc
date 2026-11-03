import numpy as np

a = np.array([10, 20, 30])
i = nondet_int()
__ESBMC_assume(i == -1)

assert a[i] == 20
