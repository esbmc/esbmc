import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)
lst = t.tolist()

a[1][0] = 9

assert a[1][0] == 9
assert lst[0][1] == 3
