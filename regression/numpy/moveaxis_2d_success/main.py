import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.moveaxis(a, 0, 1)

t[0][1] = 7

assert a[1][0] == 7
assert t[0][1] == 7
