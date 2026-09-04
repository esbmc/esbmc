import numpy as np

a = np.array([1, 3, 5, 7])
i = np.searchsorted(a, 4)

assert i == 2
