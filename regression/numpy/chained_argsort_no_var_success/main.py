import numpy as np

a = np.array([3, 1, 2]).argsort()

assert a[0] == 1
assert a[1] == 2
assert a[2] == 0
