import numpy as np

a = np.array([[1, 2], [3, 4]])
r = a[(1,)]

assert r[0] == 3
assert r[1] == 4
