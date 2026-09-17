import numpy as np

a = np.array([[1, 2], [3, 4]])
r = a.ravel()
a[1][1] = 8

assert r[3] == 8
