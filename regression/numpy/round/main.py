import numpy as np

assert np.round(1.2) == 1.0
assert np.round(1.8) == 2.0
assert np.round(-1.5) == -2.0
a = np.round(-1.8)
assert a == -2.0
