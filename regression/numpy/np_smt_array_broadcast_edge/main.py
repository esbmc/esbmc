import numpy as np

a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
c = np.add(a, b)

assert c[0] == 2
assert c[1] == 4
assert c[2] == 6
