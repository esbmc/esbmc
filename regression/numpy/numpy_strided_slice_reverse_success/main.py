import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = a[::-1]

assert b[0] == 5
assert b[4] == 1
