import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
k = np.copy(v)
k[0][0] = 99
assert v[0][0] == 5
assert c[1][0][0] == 5
assert k[1][1] == 8
