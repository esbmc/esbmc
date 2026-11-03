import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
s = a[1:, :, 1]
assert s.shape == (1, 2)
assert s[0][1] == 8
