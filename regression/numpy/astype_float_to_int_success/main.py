import numpy as np

a = np.array([1.9, 2.5, -1.9])
b = a.astype(int)

assert b[0] == 1
assert b[1] == 2
assert b[2] == -1
