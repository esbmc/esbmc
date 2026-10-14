import numpy as np

a = np.array([[1], [2], [3]])
t = np.transpose(a)

t[0][2] = 6

assert a[2][0] == 6
assert t.shape[0] == 1
assert t.shape[1] == 3
