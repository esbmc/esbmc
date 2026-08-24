import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
np.fill_diagonal(a, 9)

assert a[0][0] == 9
assert a[1][1] == 9
assert a[0][1] == 2
