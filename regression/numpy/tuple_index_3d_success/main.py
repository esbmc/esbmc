import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

assert a[2, 1, 0] == 11
assert a[0, 1, 1] == 4
