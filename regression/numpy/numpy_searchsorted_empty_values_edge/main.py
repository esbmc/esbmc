import numpy as np

a = np.array([1, 3, 5])
idx = np.searchsorted(a, [])

assert len(idx) == 0
