import numpy as np

# NumPy 2 spells arcsinh, arccosh, arctanh and arctan2 as these aliases too.
x = nondet_float()
if x >= 1.0 and x <= 2.0:
    assert np.asinh(x) > 0.88 and np.asinh(x) < 1.45
    assert np.acosh(x) >= 0.0 and np.acosh(x) < 1.32
    assert np.atanh(x / 4) > 0.12 and np.atanh(x / 4) < 0.55
    assert np.atan2(x, 1.0) > 0.0 and np.atan2(x, 1.0) < 1.571
