import numpy as np

n = nondet_int()
__ESBMC_assume(n >= 1 and n <= 4)
a = np.zeros(n)

assert a[0] == 0.0
assert a[1] == 0.0
assert a[2] == 0.0
assert a[3] == 0.0
