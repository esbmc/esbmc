import numpy as np
a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
b = np.reshape(a, (2, 2, 2))
assert b[0][0][0] == 1
assert b[0][0][1] == 2
assert b[0][1][0] == 3
assert b[1][0][0] == 5
assert b[1][1][1] == 8
