import math
import numpy as np

x = 0.5
assert np.log1p(0.5) > 0.405 and np.log1p(0.5) < 0.406
assert math.log1p(x) > 0.405 and math.log1p(x) < 0.406
assert np.arctanh(x) > 0.549 and np.arctanh(x) < 0.55
assert math.acosh(x + 1.0) > 0.962 and math.acosh(x + 1.0) < 0.963
assert np.cbrt(-8.0) > -2.0001 and np.cbrt(-8.0) < -1.9999
assert math.hypot(x, 1.0) >= 1.0 and math.hypot(x, 1.0) < 1.5
assert np.expm1(1000.0) == np.inf
