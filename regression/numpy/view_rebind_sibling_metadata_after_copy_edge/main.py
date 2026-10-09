import numpy as np

a = np.array([1, 2, 3, 4])
x = a[1:3]
c = np.array(x)
y = a[1:3]
a = np.array([9, 9, 9, 9])
x[0] = 7
assert y[0] == 7
assert c[0] == 2
c[1] = 5
assert x[1] == 3
assert y[1] == 3
