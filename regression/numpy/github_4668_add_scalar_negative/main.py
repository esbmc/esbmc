import numpy as np

a = np.array([7, 8, 9])
b = a + (-2)

assert b[0] == 5
assert b[1] == 6
assert b[2] == 7
