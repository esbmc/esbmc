import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
t = np.transpose(a)

a[1][0] = 9

assert t[0][0] == 1
assert t[0][1] == 9
assert t[1][0] == 2
assert t[2][1] == 6
