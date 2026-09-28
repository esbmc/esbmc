import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a
t = np.transpose(b)

a[0][1] = 9

assert t[1][0] == 9
