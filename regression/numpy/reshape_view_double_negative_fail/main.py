import numpy as np

a = np.array([1, 2, 3, 4])
r = np.reshape(a, (-1, -1))

assert r[0][0] == 1
