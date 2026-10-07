import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
v1 = a[0]
v2 = a[0]
a = np.array([[7, 7, 7], [7, 7, 7]])
v1[0] = 9
assert v2[0] == 9
assert v2[1] == 2
assert a[0][0] == 7
