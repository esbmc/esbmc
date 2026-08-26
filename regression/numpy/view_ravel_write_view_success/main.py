import numpy as np

a = np.array([[1, 2], [3, 4]])
r = np.ravel(a)
r[2] = 9

assert a[1][0] == 9
