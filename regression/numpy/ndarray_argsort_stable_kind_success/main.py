import numpy as np

a = np.array([5, 3, 5, 1])
idx = a.argsort(kind="stable")

assert idx[0] == 3
assert idx[1] == 1
