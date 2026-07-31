import numpy as np

n = nondet_bool()
a = np.array([[1, 2], [3, 4]])
row = a[0]

if n:
    row = np.array([7, 8])

a[0][0] = 99
