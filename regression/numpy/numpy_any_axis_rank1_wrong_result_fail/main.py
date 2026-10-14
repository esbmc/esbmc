import numpy as np

a = np.array([False, True, False])

assert np.any(a, axis=0) == False
