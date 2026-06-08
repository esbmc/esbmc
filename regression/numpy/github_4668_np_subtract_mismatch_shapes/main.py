import numpy as np

a = np.array([5, 6, 7])
b = np.array([1, 2])
c = np.subtract(a, b)
assert c[0] == 4
