import numpy as np


assert np.tan(1e-12) > 0.0
assert np.arcsin(1.0) > 1.5
assert np.log(1.0) == 0.0
assert np.log2(1.0) == 0.0
assert np.log10(1.0) == 0.0
assert np.sinh(1e-12) > 0.0
assert np.cosh(0.0) == 1.0
assert np.tanh(1e-12) > 0.0
assert np.rint(1.5) == 2.0

frac, integer = np.modf(-3.5)
assert frac == -0.5
assert integer == -3.0

mantissa, exponent = np.frexp(0.0)
assert mantissa == 0.0
assert exponent == 0

assert np.copysign(0.0, -1.0) == -0.0
assert np.fmax(2.0, 2.0) == 2.0
assert np.fmin(2.0, 2.0) == 2.0
