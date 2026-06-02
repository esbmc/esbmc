import numpy as np

a = np.array([7, 8, 9])
b = a + 0

assert b[0] == 7
assert b[1] == 8
assert b[2] == 9
