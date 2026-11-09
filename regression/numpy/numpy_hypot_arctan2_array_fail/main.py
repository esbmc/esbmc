import numpy as np

x = nondet_float()
__ESBMC_assume(x >= 1.0 and x <= 2.0)
a = np.arctan2(np.array([1.0, x]), np.array([1.0, 1.0]))
assert a[1] < 0.9
