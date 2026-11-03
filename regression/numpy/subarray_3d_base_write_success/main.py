import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
row = a[1]
a[1][1][0] = 77

assert row[1][0] == 77
assert a[1][1][0] == 77
assert row[0][0] == 5
