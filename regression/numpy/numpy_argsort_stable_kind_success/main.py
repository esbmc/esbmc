import numpy as np

a = np.array([5, 3, 5, 1])
idx = np.argsort(a, kind="stable")

assert idx[0] == 3
assert idx[1] == 1
assert idx[2] == 0
assert idx[3] == 2
