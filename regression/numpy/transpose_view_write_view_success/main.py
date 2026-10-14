import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
t = np.transpose(a)

t[2][0] = 9

assert a[0][2] == 9
assert t[2][0] == 9
