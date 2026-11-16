import math
import numpy as np

x = nondet_float()
__ESBMC_assume(x >= 0.5 and x <= 2.0)
assert np.isfinite(np.log1p(x))
assert not np.isnan(np.expm1(x))
assert not np.isinf(np.cbrt(x))
assert np.isinf(np.expm1(710.0))
assert np.isnan(np.arccosh(0.5))
assert not np.isfinite(np.log1p(-1.0))
a = np.array([x, 1.0])
f = np.isfinite(a)
assert f[0] and f[1]
n = np.isnan(np.arctanh(np.array([0.5, 2.0])))
assert not n[0] and n[1]
i = np.isinf(np.array([[1.0, 2.0], [3.0, 4.0]]))
assert not i[1][1]
v = np.isfinite(a[1:])
assert v[0]
