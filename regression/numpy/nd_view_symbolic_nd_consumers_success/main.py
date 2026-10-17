import numpy as np

a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
st = nondet_int()
__ESBMC_assume(st == 2)
r = a[::st]
assert r.sum() == 24
t = r.tolist()
assert t[1][2] == 8
k = np.copy(r)
k[0][0] = 50
assert a[0][0] == 0
assert k[1][2] == 8
assert k.shape == (2, 3)
