import numpy as np

a = np.array([1, 2, 3])
empty = a[1:1]

assert empty.shape[0] == 0
assert empty.ndim == 1
