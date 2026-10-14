import numpy as np

a = np.array([[1, 2], [3, 4]])
v = np.swapaxes(a, 0, 1)
v[0][0] = 10

assert a[0][0] == 10
