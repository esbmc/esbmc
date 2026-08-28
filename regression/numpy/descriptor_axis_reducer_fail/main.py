import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)

np.sum(t, axis=0)
