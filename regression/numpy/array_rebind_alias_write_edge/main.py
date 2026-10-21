import numpy as np

a = np.array([1, 2, 3])
b = a
a[0] = 5
b[1] = 6
a = np.array([9, 9, 9])
assert b[0] == 5 and b[1] == 6 and b[2] == 3
b[2] = 4
assert a[2] == 9
