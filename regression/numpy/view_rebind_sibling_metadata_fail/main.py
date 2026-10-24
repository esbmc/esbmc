import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
r1 = a[0]
r2 = a[0]
a = np.array([[7, 7], [7, 7]])
r1[0] = 9
assert r2[0] == 9
assert len(r2) == 3
assert len(r2) == len(a[0])
