import numpy as np

a = np.array([1, 2, 3])
v = np.broadcast_to(a, (2, 2))

assert v[0][0] == 1
