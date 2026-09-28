import numpy as np

a = np.array([2, 1, 2, 1])
idx = a.argsort()

assert idx[0] == 1
assert idx[1] == 3
assert idx[2] == 0
assert idx[3] == 2
