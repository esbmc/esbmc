import numpy as np

a = np.array([1, 2, 3, 4])
x = a[1:3]
y = a[1:3]
a = np.array([5, 5, 5, 5])
w = a[0:2]
v = a[0:2]
a = np.array([8, 8, 8, 8])
x[0] = 7
w[0] = 6
assert y[0] == 7
assert v[0] == 6
assert x[1] == 3
assert w[1] == 5
assert a[0] == 8
