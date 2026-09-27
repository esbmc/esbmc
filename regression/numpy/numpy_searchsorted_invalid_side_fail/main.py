import numpy as np

a = np.array([1, 3, 5])
idx = np.searchsorted(a, 3, side="middle")
