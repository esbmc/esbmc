import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
m = np.moveaxis(a, 0, 2)
assert m[0][0][1] == 1
