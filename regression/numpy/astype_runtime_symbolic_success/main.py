import numpy as np

x = nondet_int()
__ESBMC_assume(x >= 1 and x <= 10)
a = np.array([x, x + 1])
b = a.astype(np.float64)

assert b[0] == float(x)
assert b[1] == float(x + 1)
