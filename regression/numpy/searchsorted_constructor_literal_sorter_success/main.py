import numpy as np

idx = np.searchsorted(np.array([5, 1, 3]), 2, sorter=[1, 2, 0])

assert idx == 1
