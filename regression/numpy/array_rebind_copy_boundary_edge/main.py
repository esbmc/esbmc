import numpy as np

a = np.array([1, 2, 3])
x = a[0:2]
b = a.copy()
a = np.array([9, 9, 9])
x[0] = 7
assert b[0] == 1
assert a[0] == 9
assert x[0] == 7
