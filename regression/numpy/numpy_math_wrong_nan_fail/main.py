import numpy as np

x = nondet_float()
__ESBMC_assume(x == -2.0)
r = np.log1p(x)
assert r == r
