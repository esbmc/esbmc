import numpy as np

# Basic rounding
assert np.round(1.2) == 1.0
assert np.round(1.8) == 2.0
assert np.round(-1.5) == -2.0
a = np.round(-1.8)
assert a == -2.0

# Very small numbers
assert np.round(0.0000001) == 0.0
assert np.round(-0.0000001) == 0.0

# Large numbers
assert np.round(1234567.89) == 1234568.0
assert np.round(-1234567.89) == -1234568.0

# Already rounded
assert np.round(10.0) == 10.0
assert np.round(-10.0) == -10.0

