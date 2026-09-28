import numpy as np

a = np.array([[1, 2], [3, 4]])
lst = a.tolist()

assert lst[0][0] == 1
assert lst[0][1] == 2
assert lst[1][0] == 3
assert lst[1][1] == 4
