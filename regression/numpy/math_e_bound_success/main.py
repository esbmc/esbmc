import math
import numpy as np

d = np.exp(1.0) - np.e
assert d > -4e-15 and d < 4e-15
assert math.exp(1.0) == math.e
