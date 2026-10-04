import numpy as np

a = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
b = a[1:3]
b[0][1] = 99
assert a[1][1] == 3
assert b.shape == (2, 2)
