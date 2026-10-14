import numpy as np

a = np.array([1, 2])
a = np.array([[1, 2], [3, 4]])

b = np.broadcast_to(a, (2, 2))
assert b[1][1] == 4

