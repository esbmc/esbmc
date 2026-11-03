import numpy as np

assert np.round(2.675, 2) == 2.68
assert np.round(2.685, 2) == 2.68
assert np.round(0.5) == 0.0
assert np.round(1.5) == 2.0
assert np.round(12345, -2) == 12300
x = 2.5
assert np.round(x) == 2.0
assert np.round(x, 0) == 2.0
