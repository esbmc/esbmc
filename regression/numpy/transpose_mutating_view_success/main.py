import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.transpose(a)

b[0][1] = 9

assert a[1][0] == 9
