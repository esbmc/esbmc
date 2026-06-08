import numpy as np

assert np.sin(1e-12) > 0.0
assert np.sin(1e-12) < 1e-10
assert np.cos(1e-12) > 0.9999999999
