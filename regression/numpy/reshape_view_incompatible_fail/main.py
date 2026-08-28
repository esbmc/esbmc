import numpy as np

a = np.array([1, 2, 3, 4])
r = np.reshape(a, (3, 2))

assert r[0][0] == 1
