import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
b = a[::2, :]

assert b[0][0] == 1
assert b[0][1] == 2
assert b[1][0] == 5
assert b[1][1] == 6
