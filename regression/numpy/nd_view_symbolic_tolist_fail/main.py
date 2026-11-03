import numpy as np

a = np.array([10, 20, 30, 40, 50])
st = nondet_int()
__ESBMC_assume(st == 2)
s = a[::st]
t = s.tolist()
assert len(t) == 3
assert t[1] == 20
assert t[2] == 50
