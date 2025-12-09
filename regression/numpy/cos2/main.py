import numpy as np

assert np.cos(1e-9) >= 0.999999999
assert np.cos(-3.0) >= -1.0
assert np.cos(-0.5) >= 0.8
