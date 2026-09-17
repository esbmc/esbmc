import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.moveaxis(a, -2, -1)

t[0][1] = 5

assert a[1][0] == 5
assert t[0][1] == 5
