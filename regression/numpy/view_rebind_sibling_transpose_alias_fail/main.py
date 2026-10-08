import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
t = np.transpose(a)
r = a[0]
a = np.array([[7, 7, 7], [7, 7, 7]])
r[1] = 9
assert t[1][0] == 2
