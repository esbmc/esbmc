import numpy as np

a = np.array([3, 1, 2])
idx = np.argsort(a, 0)

assert idx[0] == 1
