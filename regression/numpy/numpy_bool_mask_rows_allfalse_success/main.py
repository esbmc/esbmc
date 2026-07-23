import numpy as np

# an all-false symbolic mask now builds a result with a runtime
# count of zero (see numpy-architecture-decisions.md).
n = nondet_bool()
__ESBMC_assume(not n)

a = np.array([[1, 2], [3, 4]])
mask = np.array([n, n])
b = a[mask]
