import math
import numpy as np

s = np.sin(-0.0)
t = np.tanh(-0.0)
assert s == 0.0
assert math.copysign(1.0, s) == -1.0
assert math.copysign(1.0, t) == -1.0
assert np.cosh(-0.0) == 1.0
assert np.arccos(1.0) == 0.0
