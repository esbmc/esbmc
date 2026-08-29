import numpy as np

a = np.array([[1, 2], [3, 4]])
v = a.transpose()

assert v[0][0] == 1
assert v[0][1] == 3
assert v[1][0] == 2
assert v[1][1] == 4
