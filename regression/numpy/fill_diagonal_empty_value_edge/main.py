import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.fill_diagonal(a, [])

assert a[0][0] == 1
assert a[1][1] == 5
assert a[2][2] == 9
