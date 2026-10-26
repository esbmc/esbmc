import numpy as np

start = nondet_int()
stop = nondet_int()
__ESBMC_assume(start == 1)
__ESBMC_assume(stop == 3)
a = np.array([10, 20, 30, 40])
s = a[start:stop]
assert len(s) == 2
assert s[0] == 20
assert s[1] == 30
