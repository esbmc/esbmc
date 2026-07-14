import numpy as np

x = nondet_int()
y = nondet_int()
n = nondet_bool()
__ESBMC_assume(n == (x > y))
__ESBMC_assume(x > y)

a = np.array([[5, 6], [7, 8]])
mask = np.array([n, not n])
b = a[mask]

assert b[0][0] == 5
assert b[0][1] == 6
