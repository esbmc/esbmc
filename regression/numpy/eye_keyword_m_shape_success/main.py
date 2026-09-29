import numpy as np

a = np.eye(3, M=4)
assert a.shape[0] == 3
assert a.shape[1] == 4
assert a.size == 12
assert a[0][3] == 0
