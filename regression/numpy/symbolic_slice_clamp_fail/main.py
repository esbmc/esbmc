import numpy as np

a = np.array([10, 20, 30, 40])
i = nondet_int()
j = nondet_int()
__ESBMC_assume(i == -2)
__ESBMC_assume(j == 9)
s = a[i:j]
assert len(s) == 2
assert s[0] == 20
assert s[1] == 40
