import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

assert a[0, 0, 0] == 1
assert a[1, 1, 1] == 8
