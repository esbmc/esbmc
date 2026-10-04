import numpy as np

a = np.array([10, 20, 30, 40, 50])
st = nondet_int()
__ESBMC_assume(st == 2)
s = a[::st]
assert s.shape[0] == 2
assert s.size == 3
assert s.ndim == 1
