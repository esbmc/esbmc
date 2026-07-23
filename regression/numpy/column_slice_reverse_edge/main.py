import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
b = a[:, 3:0:-1]

assert b[0][0] == 4
assert b[0][1] == 3
assert b[0][2] == 2
assert b[1][0] == 8
assert b[1][1] == 7
assert b[1][2] == 6
