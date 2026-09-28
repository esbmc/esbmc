import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)

assert t[0][0] == 1
assert t[0][1] == 3
assert t[1][0] == 2
assert t[1][1] == 4
