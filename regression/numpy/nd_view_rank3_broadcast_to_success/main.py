import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
b = np.broadcast_to(v, (3, 2, 2))
assert b[2][1][1] == 8
assert b.shape == (3, 2, 2)
c[1][1][1] = 80
assert b[0][1][1] == 80
