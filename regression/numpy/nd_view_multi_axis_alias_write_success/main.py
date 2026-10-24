import numpy as np

a = np.zeros((2, 3, 4))
b = a[:, :, 0]
b[1][2] = 7
assert a[1][2][0] == 7
assert b.shape == (2, 3)
