import numpy as np

assert np.copysign(1.0, -2.0) == -1.0
assert np.copysign(-1.0, 2.0) == 1.0
