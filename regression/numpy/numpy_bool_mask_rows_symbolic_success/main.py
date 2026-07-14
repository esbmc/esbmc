import numpy as np

n = nondet_bool()
__ESBMC_assume(n)

a = np.array([[1, 2], [3, 4]])
mask = np.array([n, not n])
b = a[mask]

assert b[0][0] == 1
assert b[0][1] == 2
