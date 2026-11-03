import numpy as np

a = np.array([1, 2, 3, 4])
x = a[1:3]
b = a
c = a
a = np.array([9, 9, 9, 9])
x[0] = 7
b[2] = 8
assert b[1] == 7 and c[1] == 7
assert x[1] == 8 and c[2] == 8
assert a[1] == 9 and a[2] == 9
