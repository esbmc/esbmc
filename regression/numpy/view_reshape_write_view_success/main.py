import numpy as np

a = np.array([1, 2, 3, 4])
b = np.reshape(a, (2, 2))
b[0][0] = 99

assert a[0] == 99
