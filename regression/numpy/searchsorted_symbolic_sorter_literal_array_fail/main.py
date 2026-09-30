import numpy as np

a = np.array([5, 1, 3])
x = nondet_int()
idx = np.searchsorted(a, 2, sorter=[1, x, 0])
