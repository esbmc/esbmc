import numpy as np

a = np.array([3, 1])
a = np.array([0, 1])
idx = np.argsort(a)

assert idx[0] == 0
