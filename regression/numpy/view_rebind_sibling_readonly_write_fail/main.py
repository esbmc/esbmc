import numpy as np

a = np.array([[1, 2], [3, 4]])
d = np.diagonal(a)
s = a[1]
a = np.array([[7, 7], [7, 7]])
s[1] = 9
d[0] = 5
