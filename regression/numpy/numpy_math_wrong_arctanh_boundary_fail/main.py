import numpy as np

x = nondet_float()
__ESBMC_assume(x == 1.0)
assert np.isfinite(np.arctanh(x))
