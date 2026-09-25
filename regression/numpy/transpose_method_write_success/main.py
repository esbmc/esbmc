import numpy as np

a = np.array([[1, 2], [3, 4]])
t = a.transpose()

t[0][1] = 8

assert a[1][0] == 8
assert t[0][1] == 8
