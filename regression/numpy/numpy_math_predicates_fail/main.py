import numpy as np

x = nondet_float()
__ESBMC_assume(x >= 700.0 and x <= 720.0)
assert np.isfinite(np.expm1(x))
