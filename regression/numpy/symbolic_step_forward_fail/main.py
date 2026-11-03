import numpy as np

a = np.array([10, 20, 30, 40, 50, 60])
st = nondet_int()
__ESBMC_assume(st >= 1 and st <= 3)
s = a[::st]
n = len(s)
assert n == (5 + st) // st
assert s[0] == 20
assert s[n - 1] == a[(n - 1) * st]
assert s[-1] == a[(n - 1) * st]
