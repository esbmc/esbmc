import numpy as np

a = np.array([10, 20, 30, 40])
st = nondet_int()
__ESBMC_assume(st == 2)
s = a[::st]
a = np.array([1, 2, 3, 4])
assert s[1] == 30
