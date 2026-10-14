import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.swapaxes(a, -1, -2)

t[1][0] = 5

assert a[0][1] == 5
assert t[1][0] == 5
