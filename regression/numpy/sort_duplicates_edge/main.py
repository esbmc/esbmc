import numpy as np

a = np.array([2, 1, 2, 1])
b = np.sort(a)

assert b[0] == 1
assert b[1] == 1
assert b[2] == 2
assert b[3] == 2
