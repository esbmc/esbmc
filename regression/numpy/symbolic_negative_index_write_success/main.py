import numpy as np

a = np.array([10, 20, 30])
i = nondet_int()
__ESBMC_assume(i == -1)

a[i] = 99

assert a[2] == 99
