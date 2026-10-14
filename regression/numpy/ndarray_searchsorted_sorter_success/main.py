import numpy as np

a = np.array([5, 1, 3])
idx = a.searchsorted(2, sorter=a.argsort())

assert idx == 1
