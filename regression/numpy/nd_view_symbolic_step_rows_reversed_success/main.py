import numpy as np

a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
st = nondet_int()
__ESBMC_assume(st == -1)
r = a[::st]
assert r[0][0] == 9
assert r.shape[0] == 4
