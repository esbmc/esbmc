import numpy as np

a = np.array([3, 1, 2])
b = np.sort(a)

assert b[0] == 1
assert b[1] == 2
assert b[2] == 3
