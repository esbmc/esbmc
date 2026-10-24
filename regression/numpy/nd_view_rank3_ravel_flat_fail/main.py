import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
r = v.ravel()
assert r[2] == 6
assert v.flat[3] == 8
r[1] = 60
assert c[1][0][1] == 60
