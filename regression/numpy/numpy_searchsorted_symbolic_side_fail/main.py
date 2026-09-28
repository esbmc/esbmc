import numpy as np

a = np.array([1, 3, 5])
s = "left"
idx = np.searchsorted(a, 3, side=s)
