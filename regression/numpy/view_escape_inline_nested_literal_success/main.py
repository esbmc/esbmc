import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
assert [[a[0]]][0][0][1] == 2
assert [a[0], a[1]][1][2] == 6
