import numpy as np

a = np.array([1, 3, 5, 7])
x = nondet_int()
idx = np.searchsorted(a, [2, x])
