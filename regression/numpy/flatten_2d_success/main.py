import numpy as np
a = np.array([[5, 6], [7, 8]])
b = np.flatten(a)
assert b[0] == 5
assert b[1] == 6
assert b[2] == 7
assert b[3] == 8
