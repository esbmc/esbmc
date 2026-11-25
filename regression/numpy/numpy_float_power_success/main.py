import numpy as np

r = np.float_power(np.array([2, 3]), 2)
assert r[0] == 4.0 and r[1] == 9.0
assert r.dtype == np.float64
h = np.float_power(np.array([4.0, 9.0]), 0.5)
assert h[0] == 2.0 and h[1] == 3.0
assert np.float_power(2, -1) == 0.5
