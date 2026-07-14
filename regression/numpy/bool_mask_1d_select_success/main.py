import numpy as np

a = np.array([10, 20, 30])
mask = np.array([True, False, True])
b = a[mask]

assert b[0] == 10
assert b[1] == 30
