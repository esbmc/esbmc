import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
k = np.array(v)
k[1][0] = 70
assert v[1][0] == 70
assert k[0][1] == 6
assert k.shape == (2, 2)
