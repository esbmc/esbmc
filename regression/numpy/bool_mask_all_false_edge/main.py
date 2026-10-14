import numpy as np

a = np.array([1, 2, 3])
mask = np.array([False, False, False])
b = a[mask]

assert len(b) == 0
