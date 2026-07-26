import numpy as np

a = np.array([10, 20, 30])
n = nondet_bool()
mask = np.array([n, True, False])
b = a[mask]

assert b[len(b) - 1] == 20
