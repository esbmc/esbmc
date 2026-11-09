import numpy as np

x = nondet_float()
__ESBMC_assume(x >= 0.5 and x <= 2.0)
p = np.log1p(x)
k = np.log1p(x, out=None, dtype=float)
assert p > 0.4 and p < 1.1
assert k > 0.4 and k < 1.1
h = np.hypot(x, 1.0)
hk = np.hypot(x, 1.0, out=None, dtype=float)
assert h >= 1.0 and h < 2.9
assert hk >= 1.0 and hk < 2.9
assert np.isnan(np.nan, out=None)
assert not np.isfinite(np.inf, out=None)
