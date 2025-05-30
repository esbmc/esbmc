import numpy as np
b = np.transpose([[1000000, 2000000], [3000000, 4000000]])
assert b[0][0] == 1000000
assert b[0][1] == 3000000
assert b[1][0] == 2000000
assert b[1][1] == 4000000
