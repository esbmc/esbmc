import numpy as np

a = np.array([10, 20, 30, 40, 50])
st = nondet_int()
__ESBMC_assume(st == 2)
s = a[::st]
assert np.sum(s) == 90
assert s.sum() == 91
assert s.max() == 50
assert np.min(s) == 10
assert s.mean() == 30.0
assert s.any()
assert s.all()
