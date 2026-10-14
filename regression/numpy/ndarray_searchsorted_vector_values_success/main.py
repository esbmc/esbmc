import numpy as np

a = np.array([1, 3, 5, 7])
idx = a.searchsorted([2, 6])

assert idx[0] == 1
assert idx[1] == 3
