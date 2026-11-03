import numpy as np

a = np.array([1, 2, 3, 4])
x = a[1:3]
y = a[1:3]
c = x.copy()
a = np.array([9, 9, 9, 9])
y[0] = 7
assert c[0] == 2
assert x[0] == 7
c[1] = 5
assert y[1] == 3
assert x[1] == 3
