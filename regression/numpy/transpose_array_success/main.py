import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.transpose(a)

assert b[0][0] == 1
assert b[0][1] == 3
assert b[1][0] == 2
assert b[1][1] == 4
