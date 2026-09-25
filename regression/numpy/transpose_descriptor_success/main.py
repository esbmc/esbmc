import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.transpose(a)

assert b[0][0] == 1
assert b[0][1] == 4
assert b[1][0] == 2
assert b[1][1] == 5
assert b[2][0] == 3
assert b[2][1] == 6
