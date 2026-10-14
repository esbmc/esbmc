import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
t = np.transpose(a)

assert np.sum(t) == 21
