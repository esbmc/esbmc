import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
r = a.reshape(3, 2)
s = a[0]
a = np.array([[7, 7, 7], [7, 7, 7]])
s[1] = 9
assert r[0][1] == 2
