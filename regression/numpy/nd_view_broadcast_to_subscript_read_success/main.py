import numpy as np

a = np.array([1, 2, 3])
c = np.broadcast_to(a, (2, 2, 3))[1]
assert c[0][0] == 1
assert c[1][2] == 3
a[2] = 7
row = np.broadcast_to(a, (2, 3))[1]
assert row[2] == 7
