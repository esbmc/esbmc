import numpy as np
a = np.array([1, 2, 3, 4, 5, 6])
b = np.reshape(a, (2, 3))
assert b[0][0] == 1
assert b[0][2] == 3
assert b[1][0] == 4
assert b[1][2] == 6
