import numpy as np
b = np.transpose([[1], [2], [3], [4], [5]])
assert b[0][0] == 1
assert b[0][1] == 2
assert b[0][2] == 3
assert b[0][3] == 4
assert b[0][4] == 5
