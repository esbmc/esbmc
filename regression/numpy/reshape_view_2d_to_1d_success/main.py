import numpy as np

a = np.array([[1, 2], [3, 4]])
r = np.reshape(a, (4,))

r[2] = 8

assert a[1][0] == 8
