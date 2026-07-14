import numpy as np

a = np.array([1, 2, 3])
b = a + 100

assert b[0] == 101
assert b[1] == 102
assert b[2] == 103
