import numpy as np

a = np.array([True, False, True])
b = a.astype(int)

assert b[0] == 1
assert b[1] == 0
assert b[2] == 1
