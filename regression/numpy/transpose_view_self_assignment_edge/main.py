import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)

a = a
a[0][1] = 8

assert t[1][0] == 8
