import numpy as np

a = np.array([[1, 2], [3, 4]])
r = np.reshape(a, (4,))

a[0][1] = 6

assert r[1] == 6
