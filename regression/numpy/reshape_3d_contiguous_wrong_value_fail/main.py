import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
r = np.reshape(a, (2, 2, 2))
assert r[1][0][1] == 7
