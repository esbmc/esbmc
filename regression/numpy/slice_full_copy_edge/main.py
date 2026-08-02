import numpy as np

a = np.array([10, 20, 30])
b = a[:]
assert b[0] == 10
assert b[1] == 20
assert b[2] == 30
