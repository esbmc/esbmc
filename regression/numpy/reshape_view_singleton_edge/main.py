import numpy as np

a = np.array([1, 2, 3, 4])
r = np.reshape(a, (1, 4))

r[0][2] = 5

assert a[2] == 5
