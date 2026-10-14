import numpy as np

assert np.exp(10.0) > np.exp(9.0)
y = np.arctan(1e20)
assert y > 1.56
assert y < 1.58
