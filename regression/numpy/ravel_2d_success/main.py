import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.ravel(a)
assert b[0] == 1
assert b[1] == 2
assert b[2] == 3
assert b[3] == 4
