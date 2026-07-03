import numpy as np

a = np.array([[1, 2], [3, 4]])
n = nondet_bool()
mask = np.array([n, not n])
x = a[mask]
