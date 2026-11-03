import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
r = v.reshape(4)
assert r[3] == 8
r[0] = 50
assert c[1][0][0] == 50
