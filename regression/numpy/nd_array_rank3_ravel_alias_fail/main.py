import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
r = c.ravel()
assert r[5] == 6
r[5] = 60
assert c[1][0][1] == 6
