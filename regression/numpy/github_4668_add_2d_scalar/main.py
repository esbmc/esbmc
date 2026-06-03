import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a + 10

assert b[0, 0] == 11
assert b[0, 1] == 12
assert b[1, 0] == 13
assert b[1, 1] == 14
