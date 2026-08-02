import numpy as np

a = np.array([[[1, 2, 3]]])

assert a[0, 0, 0] == 1
assert a[0, 0, 2] == 3
