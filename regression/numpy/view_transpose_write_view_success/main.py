import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)
t[0][0] = 10

assert a[0][0] == 10
