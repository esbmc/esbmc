import numpy as np

a = np.array([[3, 1], [4, 2]])
b = np.sort(a, axis=None)

assert b[0] == 1
assert b[1] == 2
assert b[2] == 3
assert b[3] == 4
