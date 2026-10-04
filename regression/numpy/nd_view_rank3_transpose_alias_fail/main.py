import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
t = v.T
assert t[1][0] == 6
t[0][1] = 70
assert c[1][1][0] == 7
