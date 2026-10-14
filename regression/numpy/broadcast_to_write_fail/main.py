import numpy as np

a = np.array([1, 2, 3])
v = np.broadcast_to(a, (2, 3))
v[0][1] = 7

assert a[1] == 2
