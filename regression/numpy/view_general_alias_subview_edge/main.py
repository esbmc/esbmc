import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
b = v
w = b[0:2]
w[1] = 8
assert sum(v) == 10
assert max(b) == 8
assert a[2] == 8
