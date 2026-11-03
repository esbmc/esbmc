import numpy as np

a = np.array([1, 2, 3])
c = np.broadcast_to(a, (2, 2, 3))[1]
c[0][0] = 9
