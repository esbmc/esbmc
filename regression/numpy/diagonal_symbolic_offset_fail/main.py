import numpy as np

k = nondet_int()
a = np.array([[1, 2], [3, 4]])
d = np.diagonal(a, offset=k)
