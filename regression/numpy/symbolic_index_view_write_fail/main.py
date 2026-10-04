import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
r = a[1]
c = np.zeros((2, 2, 2))
v = c[1]
i = nondet_int()
j = nondet_int()
__ESBMC_assume(i == -1)
__ESBMC_assume(j == 0)
r[i] = 99
v[i][j] = 7
assert a[1][2] == 98
assert c[1][1][0] == 7
