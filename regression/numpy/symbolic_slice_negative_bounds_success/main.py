import numpy as np

start = nondet_int()
stop = nondet_int()
__ESBMC_assume(start == -3)
__ESBMC_assume(stop == -1)
a = np.array([10, 20, 30, 40])
s = a[start:stop]
assert len(s) == 2
assert s[1] == 30
