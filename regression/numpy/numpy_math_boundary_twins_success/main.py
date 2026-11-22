import math
import numpy as np

x = nondet_float()
__ESBMC_assume(x == 0.0 and math.copysign(1.0, x) == -1.0)
assert math.copysign(1.0, np.log1p(x)) == -1.0
assert math.copysign(1.0, np.arctanh(x)) == -1.0
y = nondet_float()
__ESBMC_assume(y == -1.0)
assert np.log1p(y) == -np.inf
assert np.arctanh(y) == -np.inf
z = nondet_float()
__ESBMC_assume(z == 1.0)
assert np.arctanh(z) == np.inf
w = nondet_float()
__ESBMC_assume(w == -2.0)
assert np.isnan(np.log1p(w))
assert np.isnan(np.arctanh(w))
