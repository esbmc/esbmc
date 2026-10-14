import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.fill_diagonal(a, [10, 20, 30])

assert a[0][0] == 10
assert a[1][1] == 20
assert a[2][2] == 30
assert a[0][1] == 2
