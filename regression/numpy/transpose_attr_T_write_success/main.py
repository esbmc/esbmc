import numpy as np

a = np.array([[1, 2], [3, 4]])
t = a.T

t[1][0] = 9

assert a[0][1] == 9
assert t[1][0] == 9
