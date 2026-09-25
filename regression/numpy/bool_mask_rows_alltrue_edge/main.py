import numpy as np

# An all-true symbolic mask preserves every row, in order.
a = np.array([[1, 2], [3, 4]])
m0 = nondet_bool()
m1 = nondet_bool()
__ESBMC_assume(m0)
__ESBMC_assume(m1)
mask = np.array([m0, m1])
b = a[mask]

assert b.shape[0] == 2
assert b.shape[1] == 2
assert b[0][0] == 1
assert b[0][1] == 2
assert b[1][0] == 3
assert b[1][1] == 4
