import numpy as np

a = np.array([1, 2, 3])
mask = np.array([True, True, True])
b = a[mask]

assert b[0] == 1
assert b[1] == 2
assert b[2] == 3
