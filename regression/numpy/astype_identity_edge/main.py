import numpy as np

a = np.array([1, 2, 3])
b = a.astype(np.int64)

assert b[0] == 1
assert b[1] == 2
assert b[2] == 3
assert len(b) == len(a)
