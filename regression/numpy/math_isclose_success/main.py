import numpy as np

assert np.isclose(0.1 + 0.2, 0.3)
assert np.isclose(100.0, 100.00001)
assert np.isclose(1e-9, 0.0)
assert not np.isclose(100.0, 100.1)
