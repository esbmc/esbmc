import numpy as np

a = np.array([5])
b = np.broadcast_to(a, (2, 2))
lst = b.tolist()

a[0] = 9

assert lst[0][0] == 5
assert lst[1][1] == 5
