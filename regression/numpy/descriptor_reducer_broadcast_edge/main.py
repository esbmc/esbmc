import numpy as np

a = np.array([5])
v = np.broadcast_to(a, (2, 3))

assert np.sum(v) == 30
