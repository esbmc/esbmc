import numpy as np

a = np.array([[1, 2], [3, 4]])
d = np.diagonal(a)
s = a[1]
a = np.array([[7, 7], [7, 7]])
s[1] = 9
assert d[1] == 9
assert d[0] == 1
assert a[1][1] == 7
