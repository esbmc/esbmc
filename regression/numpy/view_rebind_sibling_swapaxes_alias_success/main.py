import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
s = np.swapaxes(a, 0, 1)
r = a[1]
a = np.array([[7, 7, 7], [7, 7, 7]])
r[2] = 9
assert s[2][1] == 9
assert s[0][0] == 1
assert a[1][2] == 7
