import numpy as np

a = np.eye(2)
assert a[0][0] == 1
assert a[0][1] == 0
assert a[1][0] == 0
assert a[1][1] == 1
