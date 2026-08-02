import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.concatenate([a, b])

assert c[0] == 1
assert c[2] == 3
assert c[3] == 4
assert c[5] == 6
