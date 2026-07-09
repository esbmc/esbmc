import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
row = a[0]
assert row[0][0] == 1
