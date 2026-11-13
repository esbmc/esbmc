import math
import numpy as np

a = np.array([3.0, 5.0])
b = np.array([4.0, 12.0])
assert abs(np.hypot(a, b)[1] - 13.0) < 1e-12
assert abs(np.arctan2(a, b)[0] - 0.6435011087932844) < 1e-12
assert abs(np.power(2.0, 0.5) - 1.4142135623730951) < 1e-12
assert np.power(np.array([2.0, 3.0]), 2.0)[1] == 9.0
assert abs(math.gamma(4.5) - 11.631728396567448) < 1e-12
