import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
i = nondet_int()
__ESBMC_assume(i >= 0 and i <= 1)
v = a[:, i:, :]
a = np.array([[[9, 9], [9, 9]], [[9, 9], [9, 9]]])
assert v[1][0][1] == 6 or v[1][0][1] == 8
