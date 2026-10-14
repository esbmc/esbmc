import numpy as np

a = np.array([1, 2])
v = np.broadcast_to(a, (2, 2))
v[0][0] = 10

assert a[0] == 1
