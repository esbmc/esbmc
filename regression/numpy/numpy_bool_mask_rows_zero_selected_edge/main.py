import numpy as np

n = nondet_bool()
__ESBMC_assume(not n)

a = np.array([[1, 2], [3, 4]])
mask = np.array([n, n])
b = a[mask]

assert True
