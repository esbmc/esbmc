import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])

assert a[[0, 2]][0][0] == 1
assert a[[0, 2]][0][1] == 2
assert a[[0, 2]][1][0] == 5
assert a[[0, 2]][1][1] == 6
