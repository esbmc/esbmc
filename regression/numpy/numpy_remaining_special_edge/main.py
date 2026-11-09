import math
import numpy as np

# numpy.* returns NaN, an infinity or a signed zero where math.* raises.
inf = float("inf")
assert np.log1p(-1.0) == -inf
assert np.arctanh(1.0) == inf and np.arctanh(-1.0) == -inf
assert np.expm1(710.0) == inf and np.expm1(-inf) == -1.0
assert np.exp2(1024.0) == inf and np.exp2(-inf) == 0.0
r = np.arccosh(0.5)
assert r != r
for zero in (np.expm1(-0.0), np.log1p(-0.0), np.arcsinh(-0.0), np.arctanh(-0.0),
             np.cbrt(-0.0)):
    assert zero == 0.0 and math.copysign(1.0, zero) == -1.0

x = nondet_float()
if x < -1.0:
    r = np.log1p(x)
    assert r != r
if x == -1.0:
    assert np.log1p(x) == -inf
