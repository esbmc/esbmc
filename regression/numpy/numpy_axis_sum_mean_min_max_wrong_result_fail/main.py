import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

s0 = np.sum(a, axis=0)

assert s0[0] == 999
