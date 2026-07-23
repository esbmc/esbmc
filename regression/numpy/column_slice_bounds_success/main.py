import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
b = a[:, 1:3:2]

assert b[0][0] == 2
assert b[1][0] == 6
