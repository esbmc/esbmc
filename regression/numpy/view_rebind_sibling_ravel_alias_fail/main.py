import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
r = a.ravel()
s = a[1]
a = np.array([[7, 7, 7], [7, 7, 7]])
s[0] = 9
assert r[3] == 4
