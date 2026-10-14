import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)
out = np.array([0])

np.sum(t, out=out)
