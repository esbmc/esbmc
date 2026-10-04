import numpy as np

a = np.array([10, 20, 30, 40, 50])
i = nondet_int()
j = nondet_int()
__ESBMC_assume(i == 0)
__ESBMC_assume(j == 5)
s = a[i:j:2]
r = s.reshape(3, 1)
r[0][0] = 99
assert a[0] == 99
assert r[2][0] == 50
f = s.flatten()
f[1] = 5
assert a[2] == 30
