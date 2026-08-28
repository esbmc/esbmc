import numpy as np

a = np.array([[1, 2], [3, 4]])
u = np.transpose(a)
t = np.transpose(u)

t[0][1] = 7

assert a[0][1] == 7
assert t[0][1] == 7
