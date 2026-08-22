import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a
row = b[0]
b[0][0] = 99

assert row[0] == 99
assert a[0][0] == 99
