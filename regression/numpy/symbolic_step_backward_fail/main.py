import numpy as np

a = np.array([10, 20, 30, 40, 50, 60])
st = nondet_int()
__ESBMC_assume(st >= -3 and st <= -1)
s = a[::st]
n = len(s)
assert n == (5 - st) // (0 - st)
assert s[0] == 50
assert s[n - 1] == a[5 + (n - 1) * st]
