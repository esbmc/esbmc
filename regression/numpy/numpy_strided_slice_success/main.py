import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = a[::2]

assert b[0] == 1
assert b[1] == 3
assert b[2] == 5
