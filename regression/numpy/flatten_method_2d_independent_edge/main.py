import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a.flatten()
b[0] = 99

assert a[0][0] == 1
assert b[0] == 99
