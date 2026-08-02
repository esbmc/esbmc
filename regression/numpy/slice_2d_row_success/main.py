import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

assert a[1, :][0] == 4
assert a[1, :][1] == 5
assert a[1, :][2] == 6
