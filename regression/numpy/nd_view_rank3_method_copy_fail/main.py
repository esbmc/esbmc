import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
k = v.copy()
k[0][1] = 60
assert c[1][0][1] == 60
assert k[0][1] == 60
