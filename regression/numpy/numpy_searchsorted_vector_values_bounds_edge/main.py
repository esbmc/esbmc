import numpy as np

a = np.array([2, 4, 6])
idx = np.searchsorted(a, [0, 100])

assert idx[0] == 0
assert idx[1] == 3
