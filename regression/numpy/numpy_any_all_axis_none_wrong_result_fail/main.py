import numpy as np

b = np.array([[False, False], [False, True]])

assert np.any(b, axis=None) == False
