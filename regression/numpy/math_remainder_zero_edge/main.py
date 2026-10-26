import numpy as np

r = np.remainder(7.0, 0.0)
assert r != r
assert np.remainder(0.0, 5.0) == 0.0
