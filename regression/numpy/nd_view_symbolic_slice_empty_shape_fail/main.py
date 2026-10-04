import numpy as np

a = np.array([10, 20, 30, 40])
i = nondet_int()
j = nondet_int()
__ESBMC_assume(i == 3)
__ESBMC_assume(j == 1)
s = a[i:j]
assert len(s) == 0
assert s.shape == (1,)
