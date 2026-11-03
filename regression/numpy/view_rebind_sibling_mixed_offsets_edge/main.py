import numpy as np

a = np.array([1, 2, 3, 4, 5])
x = a[0:3]
y = a[2:5]
a = np.array([9, 9, 9, 9, 9])
x[2] = 7
assert y[0] == 7
y[2] = 8
assert x[1] == 2
assert y[1] == 4
assert a[4] == 9
