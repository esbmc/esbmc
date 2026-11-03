import numpy as np

a = np.array([10, 20, 30, 40, 50])
st = nondet_int()
__ESBMC_assume(st == 2)
s = a[::st]
s[1] = 99
assert a[2] == 99
a[4] = 7
assert s[2] == 7
nst = nondet_int()
__ESBMC_assume(nst == -1)
r = a[::nst]
r[0] = 5
assert a[4] == 5
