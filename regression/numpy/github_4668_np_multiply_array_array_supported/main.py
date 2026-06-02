import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])
c = np.multiply(a, b)
assert c[1] == 8
