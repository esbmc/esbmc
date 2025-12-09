import numpy as np
assert np.isclose(0.1 + 0.2, 0.3)  # True
assert np.isclose(100.0, 100.00001)  # True
assert not np.isclose(100.0, 100.1)  # False

