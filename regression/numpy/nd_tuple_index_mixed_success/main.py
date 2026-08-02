import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

i = 1
j = 0
assert a[i, j, 1] == 6
