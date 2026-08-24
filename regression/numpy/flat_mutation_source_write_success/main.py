import numpy as np

a = np.array([[1, 2], [3, 4]])
a[1][1] = 8

assert a.flat[3] == 8
