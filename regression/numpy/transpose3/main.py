import numpy as np
b = np.transpose([[0, 1], [2, 0]])
assert b[0][0] == 0
assert b[0][1] == 2
assert b[1][0] == 1
assert b[1][1] == 0
