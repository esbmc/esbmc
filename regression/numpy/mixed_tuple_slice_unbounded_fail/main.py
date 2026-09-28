import numpy as np

n = nondet_int()
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = a[0:n, 0, 0]
