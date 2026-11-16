import numpy as np

assert np.expm1(0.0, out=None) == 0.0 and np.log1p(0.0, dtype=float) == 0.0
x = nondet_float()
if x >= 1.0 and x <= 2.0:
    assert np.expm1(x, out=None) > 1.71 and np.expm1(x, dtype=float) < 6.39
    assert np.hypot(x, 1.0, out=None) >= 1.0 and np.hypot(x, 1.0, dtype=float) <= 3.0
