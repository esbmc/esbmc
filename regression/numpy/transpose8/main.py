import numpy as np
b = np.transpose([[-1, 2], [3, -4]])
assert b[0][0] == -1
assert b[1][1] == -4
