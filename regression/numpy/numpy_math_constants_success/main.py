import math
import numpy as np

x = nondet_float()
__ESBMC_assume(x >= 0.5 and x <= 2.0)
assert np.inf == math.inf
assert -np.inf < x and x < np.inf
assert np.isinf(np.inf) and np.isinf(-np.inf)
assert np.isnan(np.nan) and math.isnan(np.nan)
assert math.isinf(np.inf) and not math.isfinite(np.inf)
assert np.isnan(math.nan) and np.isinf(math.inf)
assert np.nan != np.nan
assert np.expm1(np.inf) == np.inf
assert np.isnan(np.log1p(np.nan))
assert np.arctan2(1.0, np.inf) == 0.0
