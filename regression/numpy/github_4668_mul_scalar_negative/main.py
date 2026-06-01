import numpy as np

a = np.array([2, 3, 4])
b = a * (-2)

assert b[0] == -4
assert b[1] == -6
assert b[2] == -8
