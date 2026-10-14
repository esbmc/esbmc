import numpy as np

a = np.array([[1, 2], [3, 4]])
t = a.T
b = np.array(t)

a[1][0] = 9
b[0][1] = 8

assert a[1][0] == 9
assert b[0][1] == 8
assert b[1][0] == 2
