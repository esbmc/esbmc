import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
col = a[:, 1]
s = np.array([10, 20, 30, 40])[1:]
rs = np.arange(6).reshape(2, 3)
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
i = nondet_int()
j = nondet_int()
__ESBMC_assume(i >= -2 and i < 2)
__ESBMC_assume(j >= -3 and j < 3)

assert col[i] == (2 if i == 0 or i == -2 else 5)
assert s[j] == (20 if j == 0 or j == -3 else 30 if j == 1 or j == -2 else 40)
ii = i + 2 if i < 0 else i
jj = j + 3 if j < 0 else j
assert rs[i][j] == ii * 3 + jj
assert v[i][0] == (5 if i == 0 or i == -2 else 7)
