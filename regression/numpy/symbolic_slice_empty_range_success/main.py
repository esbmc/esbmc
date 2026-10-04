import numpy as np

a = np.array([10, 20, 30, 40])
i = nondet_int()
j = nondet_int()
__ESBMC_assume(i >= 0 and i <= 4)
__ESBMC_assume(j >= 0 and j <= 4)
__ESBMC_assume(i > j)
s = a[i:j]
assert len(s) == 0
