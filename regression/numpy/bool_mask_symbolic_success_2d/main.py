import numpy as np

a = np.array([[10, 20, 30], [40, 50, 60]])
n = nondet_bool()
mask = np.array([n, True, False])
b = a[1][mask]

assert b[len(b) - 1] == 50
