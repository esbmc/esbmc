import numpy as np

b = np.array([[False, False], [False, True]])

assert np.any(b, axis=None) == True
assert np.all(b, axis=None) == False
