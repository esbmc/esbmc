import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.broadcast_to(c, (3, 2, 2, 2))
b[0][0][0][0] = 9
