import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.moveaxis(a, 1, 1)

t[0][1] = 6

assert a[0][1] == 6
assert t[0][1] == 6
