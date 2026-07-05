import numpy as np


assert np.tan(0.0) == 0.0
assert np.arcsin(0.0) == 0.0
assert np.log(1.0) == 0.0
assert np.log2(8.0) == 3.0
assert np.log10(1000.0) > 2.99
assert np.log10(1000.0) < 3.01
assert np.sinh(0.0) == 0.0
assert np.cosh(0.0) == 1.0
assert np.tanh(0.0) == 0.0
assert np.rint(1.2) == 1.0

frac, integer = np.modf(3.5)
assert frac == 0.5
assert integer == 3.0

mantissa, exponent = np.frexp(8.0)
assert mantissa == 0.5
assert exponent == 4

assert np.isclose(0.1 + 0.2, 0.3)
assert np.copysign(1.0, -2.0) == -1.0
assert np.fmax(1.0, 2.0) == 2.0
assert np.fmin(1.0, 2.0) == 1.0
