import numpy as np

a = np.array([5, 1, 3])
idx = np.searchsorted(a, [2, 4], sorter=np.argsort(a))

assert idx[0] == 1
assert idx[1] == 2
