import numpy as np

a = np.array([10, 20, 30, 40, 50])
i = nondet_int()
__ESBMC_assume(i == 1)
s = a[i:]
r = s.reshape(3, 1)
