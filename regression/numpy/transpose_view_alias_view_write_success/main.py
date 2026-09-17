import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a
t = np.transpose(b)

t[1][0] = 9

assert a[0][1] == 9
assert b[0][1] == 9
assert t[1][0] == 9
