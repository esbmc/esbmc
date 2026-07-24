import numpy as np

pos = np.searchsorted(np.array([3, 1, 2]), 2)

assert pos == 1
