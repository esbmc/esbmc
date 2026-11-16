import numpy as np

x = nondet_float()
__ESBMC_assume(x >= -1.0 and x <= 1.0)
y = np.expm1(x, where=True)
assert y >= -0.7 and y <= 1.8
z = np.sin(0.5, where=1)
assert z > 0.47 and z < 0.48
w = np.isnan(x, where=True, out=None)
assert not w
