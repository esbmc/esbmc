import numpy as np

a = np.array([5, 1, 3])
idx = np.searchsorted(a, 2, sorter=np.argsort(a))

assert idx == 1
