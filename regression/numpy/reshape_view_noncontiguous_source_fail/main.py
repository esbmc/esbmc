import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)
r = np.reshape(t, (4,))

assert r[0] == 1
