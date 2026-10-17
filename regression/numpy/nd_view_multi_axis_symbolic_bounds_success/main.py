import numpy as np

a = np.arange(24).reshape(2, 3, 4)
i = nondet_int()
__ESBMC_assume(i == 1)
b = a[i:, :, 1]
assert b.shape == (1, 3)
assert b[0][2] == 21
b[0][0] = 5
assert a[1][0][1] == 5
j = nondet_int()
__ESBMC_assume(j == 2)
c = a[:, :j, 0]
assert len(c[0]) == 2
assert c[1][1] == 16
