import numpy as np

a = np.array([[1, 2], [3, 4]])
t = a.T
i = nondet_int()
j = nondet_int()
__ESBMC_assume(i == -1)
__ESBMC_assume(j == 0)
t[i][j] = 9
assert a[1][0] == 9
