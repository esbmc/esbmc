import numpy as np

assert np.sin(0.0) == 0.0
assert np.isclose(np.sin(np.pi / 2), 1.0)
assert np.isclose(np.sin(np.pi), 0.0, atol=1e-15)
