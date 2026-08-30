import numpy as np

a = np.array([False, True, False])

assert np.any(a, axis=0) == True
assert np.all(a, axis=0) == False
