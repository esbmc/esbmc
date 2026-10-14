import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a
row = b[0]
row[0] = 9

assert a[0][0] == 9
assert b[0][0] == 9
