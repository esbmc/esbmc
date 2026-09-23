import numpy as np

a = np.ravel(np.eye(2, dtype=int))

assert a[0] == 1
assert a[1] == 0
assert a[2] == 0
assert a[3] == 1
