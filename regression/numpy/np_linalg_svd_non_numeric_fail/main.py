import numpy as np

x = nondet_int()
a = np.array([[x, 1.0], [2.0, 3.0]])
sigma = np.linalg.svd(a)
