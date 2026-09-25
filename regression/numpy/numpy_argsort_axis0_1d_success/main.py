import numpy as np

a = np.array([3, 1, 2])
idx = np.argsort(a, axis=0)

assert idx[0] == 1
assert idx[1] == 2
assert idx[2] == 0
