import numpy as np

a = np.array([10, 20, 30, 40, 50])
st = nondet_int()
__ESBMC_assume(st == 2)
s = a[::st]
assert np.sum(s) == 90
assert np.max(s) == 50
assert np.min(s) == 10
