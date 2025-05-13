import numpy as np

assert np.trunc(1.8) == 1.0
assert np.trunc(-1.8) == -1.0
a = np.trunc(10.123)
assert a != 10.1
