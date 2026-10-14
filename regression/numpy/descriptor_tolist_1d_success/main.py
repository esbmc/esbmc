import numpy as np

a = np.array([1, 2, 3, 4])
r = np.reshape(a, (4,))
lst = r.tolist()

assert lst[0] == 1
assert lst[3] == 4
