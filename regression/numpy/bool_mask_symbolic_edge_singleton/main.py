import numpy as np

a = np.array([9])
n = nondet_bool()
mask = np.array([n])
b = a[mask]

assert len(b) <= 1
