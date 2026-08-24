import numpy as np

a = np.array([[1, 2], [3, 4]])
r = a.ravel()
r[0] = 5

assert a[0][0] == 5
