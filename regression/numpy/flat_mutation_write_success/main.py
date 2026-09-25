import numpy as np

a = np.array([[1, 2], [3, 4]])
a.flat[2] = 9

assert a[1][0] == 9
