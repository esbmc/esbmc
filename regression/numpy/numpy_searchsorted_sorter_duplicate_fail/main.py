import numpy as np

a = np.array([5, 1, 3])
idx = np.searchsorted(a, 2, sorter=[1, 1, 2])
