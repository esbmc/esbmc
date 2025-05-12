import numpy as np

assert np.isclose(np.cos(0.0), 1.0)
assert np.isclose(np.cos(np.pi), -1.0)
assert np.isclose(np.cos(np.pi / 2), 0.0, atol=1e-15)
