import numpy as np

# Basic cases
assert np.fmin(1.0, 2.0) == 1.0
assert np.fmin(3.0, -1.0) == -1.0
assert np.fmin(-5.5, -2.5) == -5.5
assert np.fmin(0.0, 0.0) == 0.0

# Very small and very large numbers
assert np.fmin(1e-308, 1e+308) == 1e-308
assert np.fmin(1e+308, 1e-308) == 1e-308

