import numpy as np

n = nondet_bool()

a = np.array([[1, 2], [3, 4]])
mask = np.array([n, not n, n])
b = a[mask]
