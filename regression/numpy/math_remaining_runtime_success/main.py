import math
import numpy as np

# Bounds for x in [1, 2], from the C library's values at the ends.
x = nondet_float()
if x >= 1.0 and x <= 2.0:
    assert np.expm1(x) > 1.71 and np.expm1(x) < 6.39
    assert math.expm1(x) > 1.71 and math.expm1(x) < 6.39
    assert np.log1p(x) > 0.69 and np.log1p(x) < 1.1
    assert math.log1p(x) > 0.69 and math.log1p(x) < 1.1
    assert np.exp2(x) > 1.99 and np.exp2(x) < 4.01
    assert math.exp2(x) > 1.99 and math.exp2(x) < 4.01
    assert np.arcsinh(x) > 0.88 and np.arcsinh(x) < 1.45
    assert math.asinh(x) > 0.88 and math.asinh(x) < 1.45
    assert np.arccosh(x) >= 0.0 and np.arccosh(x) < 1.32
    assert math.acosh(x) >= 0.0 and math.acosh(x) < 1.32
    assert np.arctanh(x / 4) > 0.12 and np.arctanh(x / 4) < 0.55
    assert math.atanh(x / 4) > 0.12 and math.atanh(x / 4) < 0.55
    assert np.cbrt(x) > 0.99 and np.cbrt(x) < 1.26
    assert math.cbrt(x) > 0.99 and math.cbrt(x) < 1.26
