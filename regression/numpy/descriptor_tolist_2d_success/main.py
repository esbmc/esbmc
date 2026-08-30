import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)
lst = t.tolist()

assert lst[0][1] == 3
assert lst[1][0] == 2
