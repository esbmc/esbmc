import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a
row = b[0]
a = np.array([[9, 8], [7, 6]])

assert row[0] == 1
assert b[0][0] == 1
