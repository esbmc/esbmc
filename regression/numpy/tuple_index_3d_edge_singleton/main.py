import numpy as np

a = np.array([[[7], [8]]])

assert a[0, 0, 0] == 7
assert a[0, 1, 0] == 8
