import numpy as np

a = np.eye(2).ravel()

assert a[0] == 1
assert a[1] == 0
assert a[2] == 0
assert a[3] == 1
