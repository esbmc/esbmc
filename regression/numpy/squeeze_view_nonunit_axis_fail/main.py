import numpy as np

a = np.array([[1], [2]])
v = np.squeeze(a, 0)

assert v[0] == 1
