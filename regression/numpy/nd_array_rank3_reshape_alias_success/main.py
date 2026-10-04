import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
r = c.reshape(8)
r[7] = 80
assert c[1][1][1] == 80
q = c.reshape(2, 4)
assert q[1][3] == 80
q[0][0] = 10
assert c[0][0][0] == 10
