import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
t = a.T
assert t[2][1] == 6
t[0][1] = 40
assert a[1][0] == 40
a[0][2] = 30
assert t[2][0] == 30
assert t.shape == (3, 2)
