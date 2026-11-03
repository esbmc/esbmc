import numpy as np

a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
st = nondet_int()
__ESBMC_assume(st == 2)
r = a[::st]
assert r.shape == (2, 3)
assert r[1][2] == 11
r[1][0] = 77
assert a[2][0] == 77
