import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
s = np.swapaxes(a, 0, 2)
assert s[1][0][0] == 1
