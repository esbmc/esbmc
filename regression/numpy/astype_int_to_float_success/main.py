import numpy as np

a = np.array([1, 2, 3])
b = a.astype(np.float64)

assert b[0] == 1.0
assert b[1] == 2.0
assert b[2] == 3.0
