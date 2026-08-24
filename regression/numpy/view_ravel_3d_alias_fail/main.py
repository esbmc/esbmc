import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = a.ravel()
a[0][0][0] = 99

assert b[0] == 1
