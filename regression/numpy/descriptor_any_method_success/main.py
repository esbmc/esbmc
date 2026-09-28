import numpy as np

a = np.array([False, False, True, False])
t = np.reshape(a, (2, 2))

assert t.any()
