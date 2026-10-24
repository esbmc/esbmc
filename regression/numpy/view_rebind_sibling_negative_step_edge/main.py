import numpy as np

a = np.array([1, 2, 3, 4])
x = a[::-1]
y = a[3:0:-1]
a = np.array([9, 9, 9, 9])
x[0] = 7
assert y[0] == 7
assert y[1] == 3
assert x[3] == 1
