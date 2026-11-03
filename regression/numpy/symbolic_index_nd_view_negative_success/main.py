import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
w = np.array([[1, 2, 3], [4, 5, 6]])
r = w[1]
t = w.T
i = nondet_int()
j = nondet_int()
k = nondet_int()
__ESBMC_assume(i == -1)
__ESBMC_assume(j == -1)
__ESBMC_assume(k == -2)

assert v[i][j] == 8
assert r[i] == 6
assert t[i][j] == 6
assert c[i][j][k] == 7
