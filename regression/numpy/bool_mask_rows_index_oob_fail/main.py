import numpy as np

# Reading b[1] when only one row was selected fails - the logical count is
# what bounds indexing, not the result's physical (worst-case) capacity.
a = np.array([[1, 2], [3, 4], [5, 6]])
m0 = nondet_bool()
m1 = nondet_bool()
m2 = nondet_bool()
__ESBMC_assume(not m0)
__ESBMC_assume(m1)
__ESBMC_assume(not m2)
mask = np.array([m0, m1, m2])
b = a[mask]
x = b[1][0]
