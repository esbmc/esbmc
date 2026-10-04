import numpy as np

a = np.array([10, 20, 30, 40, 50])
st = nondet_int()
__ESBMC_assume(st == 2)
s = a[::st]
k = np.copy(s)
k[0] = 99
assert a[0] == 99
assert k[2] == 50
assert len(k) == 3
assert np.array(s)[1] == 30
assert s.copy()[2] == 50
