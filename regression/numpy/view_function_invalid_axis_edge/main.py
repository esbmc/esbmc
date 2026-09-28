import numpy as np

a = np.array([[1, 2], [3, 4]])
v = np.swapaxes(a, 0, 3)

assert v[0][0] == 1
