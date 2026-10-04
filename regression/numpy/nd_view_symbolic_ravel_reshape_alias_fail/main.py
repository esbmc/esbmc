import numpy as np

a = np.array([10, 20, 30, 40, 50])
i = nondet_int()
j = nondet_int()
__ESBMC_assume(i == 1)
__ESBMC_assume(j == 4)
s = a[i:j]
r = s.reshape(3, 1)
assert r.shape == (3, 1)
r[1][0] = 99
assert a[2] == 30
q = np.ravel(s)
q[0] = 7
assert a[1] == 7
