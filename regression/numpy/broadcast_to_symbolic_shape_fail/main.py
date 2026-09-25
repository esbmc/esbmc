import numpy as np

a = np.array([1, 2, 3])
v = np.broadcast_to(a, (len(a), 3))

assert v[0][0] == 1
