import numpy as np

a = np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]])
b = a[:, ::2]
r = b.reshape(12)
r[0] = 55
assert a[0][0] == 55
assert r[11] == 22
