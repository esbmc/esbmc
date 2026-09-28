import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
s = np.sum(a, axis=None)

assert s == 21
