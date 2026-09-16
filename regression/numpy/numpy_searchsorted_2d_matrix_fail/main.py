import numpy as np

a = np.array([[1, 3, 5], [2, 4, 6]])
# Trying to searchsorted on entire 2D matrix - should fail
idx = np.searchsorted(a, 4)

assert idx[0] == 1
