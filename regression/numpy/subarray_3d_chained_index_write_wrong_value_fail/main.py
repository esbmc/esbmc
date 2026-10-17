import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
row = a[1][0]
row[1] = 99

assert a[1][0][1] == 6
