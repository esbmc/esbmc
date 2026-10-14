import numpy as np

a = np.array([1, 3, 5, 7])
i = a.searchsorted(4)

assert i == 2
