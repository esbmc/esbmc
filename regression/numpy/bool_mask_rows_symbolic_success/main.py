import numpy as np

# a[mask] with a symbolic mask preserves row order.
a = np.array([[1, 2], [3, 4], [5, 6]])
m0 = nondet_bool()
m1 = nondet_bool()
m2 = nondet_bool()
__ESBMC_assume(not m0)
__ESBMC_assume(m1)
__ESBMC_assume(m2)
mask = np.array([m0, m1, m2])
b = a[mask]

assert b[0][0] == 3
assert b[0][1] == 4
assert b[1][0] == 5
assert b[1][1] == 6
assert b[-1][0] == 5
assert b[-1][1] == 6
