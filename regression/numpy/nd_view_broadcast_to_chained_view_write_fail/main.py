import numpy as np

a = np.array([1, 2, 3])
b = np.broadcast_to(a, (2, 2, 3))
c = b[1]
d = c[:, ::2]
d[0][0] = 9
