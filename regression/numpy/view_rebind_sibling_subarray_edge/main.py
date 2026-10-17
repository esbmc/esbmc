import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
s1 = a[0:2, 1:3]
s2 = a[0:2, 1:3]
a = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
s1[1][0] = 50
assert s2[1][0] == 50
assert s2[0][1] == 3
assert s2.shape == (2, 2)
assert a[1][1] == 0
