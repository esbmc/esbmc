import numpy as np

assert np.remainder(7, 3) == 1
assert np.remainder(-7, 3) == 2
assert np.remainder(7, -3) == -2
assert np.remainder(7.5, 2.0) == 1.5
