import numpy as np

x = nondet_float()
__ESBMC_assume(x >= 0.5 and x <= 2.0)
assert np.nan == np.nan or x < 0.0
