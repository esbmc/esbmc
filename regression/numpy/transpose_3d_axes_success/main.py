import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
t = np.transpose(a)
assert t.shape == (2, 2, 2)
assert t[0][1][0] == 3
assert t[1][0][1] == 6
