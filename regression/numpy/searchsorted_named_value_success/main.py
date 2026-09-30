import numpy as np

v = 3
pos = np.searchsorted(np.array([1, 3, 5]), v)

assert pos == 1
