import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])
c = np.stack([a, b])

assert c[0][0] == 1
assert c[0][1] == 2
assert c[1][0] == 3
assert c[1][1] == 4
