import numpy as np

a = np.array([[1, 2], [3, 4]])
t = a.T
a[0][0] = 10

assert t[0][0] == 10
