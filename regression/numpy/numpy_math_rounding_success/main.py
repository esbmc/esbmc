import numpy as np

assert abs(np.exp(0.3) - 1.3498588075760032) < 1e-12
assert abs(np.log(2.0) - 0.6931471805599453) < 1e-12
y = np.cos([0.0, 1.0])
assert y[0] == 1.0
assert abs(y[1] - 0.5403023058681398) < 1e-12
assert np.sqrt(2.0) == 1.4142135623730951
assert np.log2(8.0) == 3.0
assert np.exp(0.0) == 1.0
assert np.log10(1.0) == 0.0
