import numpy as np

a = np.array([1, 2, 3, 4])
x = a[1:3]
y = a[1:3]
a = np.array([9, 9, 9, 9])
a[1] = 5
assert x[0] == 2
assert y[0] == 2
x[0] = 7
assert y[0] == 7
assert a[1] == 5
