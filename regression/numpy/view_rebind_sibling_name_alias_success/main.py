import numpy as np

a = np.array([1, 2, 3, 4])
x = a[1:3]
b = a
a = np.array([9, 9, 9, 9])
x[0] = 7
assert b[1] == 7
