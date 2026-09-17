import numpy as np

a = np.array([3, 1, 2])
b = np.sort(a)
b[0] = 99

assert a[0] == 3

a[0] = 77
assert b[0] == 99
