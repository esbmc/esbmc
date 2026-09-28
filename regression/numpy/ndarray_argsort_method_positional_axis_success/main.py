import numpy as np

a = np.array([3, 1, 2])
idx = a.argsort(0)

assert idx[0] == 1
