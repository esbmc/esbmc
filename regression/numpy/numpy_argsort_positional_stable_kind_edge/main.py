import numpy as np

a = np.array([3, 1, 2])
idx = np.argsort(a, None, "stable")

assert idx[0] == 1
