import numpy as np

a = np.array([[1, 2], [3, 4]])
j = nondet_int()
__ESBMC_assume(j >= 0 and j < 2)
col = a[:, j]

assert col[0] == a[0][j]
assert col[1] == a[1][j]
