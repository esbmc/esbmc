import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.swapaxes(a, 0, 2)

assert t[0][0] == 1
