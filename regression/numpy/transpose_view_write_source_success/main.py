import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])
t = np.transpose(a)

a[2][1] = 8

assert t[1][2] == 8
assert a[2][1] == 8
