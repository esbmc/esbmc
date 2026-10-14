import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
r = np.reshape(a, (2, 3))

assert np.mean(r) == 3.5
