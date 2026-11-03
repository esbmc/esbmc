import numpy as np

a = np.array([1, 2])
b = a
a[0] = 5
a = np.array([9, 9])
assert b[0] == 1
