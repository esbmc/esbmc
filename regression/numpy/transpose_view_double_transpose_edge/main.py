import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(np.transpose(a))

t[0][1] = 7

assert a[0][1] == 7
assert t[0][1] == 7
