import numpy as np

a = np.array([10, 20, 30, 40])
i = nondet_int()
j = nondet_int()
__ESBMC_assume(i == 1)
__ESBMC_assume(j == 3)
s = a[i:j]
assert np.min(s) == 20
assert np.max(s) == 30
