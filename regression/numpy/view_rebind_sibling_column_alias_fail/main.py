import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
c1 = a[:, 1]
c2 = a[:, 1]
a = np.array([[7, 7, 7], [7, 7, 7]])
c1[1] = 9
assert c2[1] == 5
