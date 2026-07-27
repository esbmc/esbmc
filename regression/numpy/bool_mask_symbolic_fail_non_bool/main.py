import numpy as np

a = np.array([1, 2, 3])
n = nondet_int()
mask = np.array([n, 0, 1])
b = a[mask]
