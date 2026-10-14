import numpy as np

# An all-false symbolic mask produces shape (0, cols).
a = np.array([[1, 2], [3, 4]])
m0 = nondet_bool()
m1 = nondet_bool()
__ESBMC_assume(not m0)
__ESBMC_assume(not m1)
mask = np.array([m0, m1])
b = a[mask]

assert b.shape[0] == 0
assert b.shape[1] == 2
assert b.ndim == 2
