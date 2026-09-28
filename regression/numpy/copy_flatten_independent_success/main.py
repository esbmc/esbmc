import numpy as np

a = np.array([[5, 6], [7, 8]])
b = np.flatten(a)

b[0] = 99

assert a[0][0] == 5
assert b[0] == 99
