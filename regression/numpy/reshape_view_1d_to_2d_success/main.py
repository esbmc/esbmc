import numpy as np

a = np.array([1, 2, 3, 4])
r = np.reshape(a, (2, 2))

a[1] = 9

assert r[0][1] == 9
