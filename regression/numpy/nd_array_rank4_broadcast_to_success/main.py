import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.broadcast_to(c, (3, 2, 2, 2))
assert b[2][1][1][1] == 8
assert b[0][0][1][0] == 3
