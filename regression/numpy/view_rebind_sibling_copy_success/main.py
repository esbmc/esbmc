import numpy as np

a = np.array([1, 2, 3, 4])
x = a[1:3]
y = a[1:3]
a = np.array([9, 9, 9, 9])
x[0] = 7
c = y.copy()
assert c[0] == 7
assert c[1] == 3
x[1] = 8
assert y[1] == 8
assert c[1] == 3
