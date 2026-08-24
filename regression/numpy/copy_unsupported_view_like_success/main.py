import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.ravel(a)

b[0] = 99

assert a[0][0] == 99
